/*
 * Copyright (c) 2026
 *
 * af_musicdamper: Sherpa-ONNX powered offline source-separation audio filter.
 *
 * This filter buffers audio in AVAudioFifo, runs chunk-based CPU inference via
 * sherpa-onnx C API, and emits the vocal stem as output. On model/inference
 * failures it safely falls back to original audio passthrough.
 */

#include <stdint.h>
#include <string.h>
#include <inttypes.h>

#include <libavutil/avassert.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/avstring.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>

#include "avfilter.h"
#include "audio.h"
#include "filters.h"
#include "formats.h"

#include "sherpa-onnx/c-api/c-api.h"

typedef struct MusicDamperContext {
    const AVClass *class;

    char *model_type;
    char *spleeter_vocals;
    char *spleeter_accompaniment;
    char *uvr_model;
    int threads;
    int chunk_size;

    const SherpaOnnxOfflineSourceSeparation *ss;

    AVAudioFifo *fifo;
    int channels;
    int sample_rate;

    float **in_chunk;
    float **orig_chunk;
    int buf_allocated_samples;

    int64_t fifo_start_pts;
    int fifo_has_pts;

    int input_eof;
    int fallback_passthrough;
} MusicDamperContext;

#define OFFSET(x) offsetof(MusicDamperContext, x)
#define A AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption musicdamper_options[] = {
    { "model_type", "Source-separation model type: spleeter or uvr",
      OFFSET(model_type), AV_OPT_TYPE_STRING, { .str = "uvr" }, 0, 0, A },
    { "spleeter_vocals", "Path to Spleeter vocals ONNX model",
      OFFSET(spleeter_vocals), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, A },
    { "spleeter_accompaniment", "Path to Spleeter accompaniment ONNX model",
      OFFSET(spleeter_accompaniment), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, A },
    { "uvr_model", "Path to UVR (MDX-Net) ONNX model",
      OFFSET(uvr_model), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, A },
    { "threads", "CPU threads for inference",
      OFFSET(threads), AV_OPT_TYPE_INT, { .i64 = 2 }, 1, 4, A },
    { "chunk_size", "Samples per separation pass",
      OFFSET(chunk_size), AV_OPT_TYPE_INT, { .i64 = 44100 }, 256, 1 << 22, A },
    { NULL }
};

AVFILTER_DEFINE_CLASS(musicdamper);

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    static const enum AVSampleFormat sample_fmts[] = {
        AV_SAMPLE_FMT_FLTP,
        AV_SAMPLE_FMT_NONE,
    };
    int ret = 0;

    ret = ff_set_common_formats_from_list2((AVFilterContext *)ctx, cfg_in, cfg_out,
                                           sample_fmts);
    if (ret < 0)
        return ret;

    ret = ff_set_common_all_channel_counts2((AVFilterContext *)ctx, cfg_in, cfg_out);
    if (ret < 0)
        return ret;

    return ff_set_common_all_samplerates2((AVFilterContext *)ctx, cfg_in, cfg_out);
}

static void update_fifo_pts_after_read(MusicDamperContext *s, AVFilterLink *outlink,
                                       int consumed)
{
    if (!s->fifo_has_pts)
        return;

    s->fifo_start_pts += av_rescale_q(consumed,
                                      (AVRational){1, outlink->sample_rate},
                                      outlink->time_base);

    if (av_audio_fifo_size(s->fifo) == 0)
        s->fifo_has_pts = 0;
}

static int ensure_chunk_buffers(MusicDamperContext *s, int nb_samples)
{
    int ret;

    if (nb_samples <= s->buf_allocated_samples)
        return 0;

    if (s->in_chunk) {
        av_freep(&s->in_chunk[0]);
        av_freep(&s->in_chunk);
    }

    if (s->orig_chunk) {
        av_freep(&s->orig_chunk[0]);
        av_freep(&s->orig_chunk);
    }

    ret = av_samples_alloc_array_and_samples((uint8_t ***)&s->in_chunk, NULL,
                                             s->channels, nb_samples,
                                             AV_SAMPLE_FMT_FLTP, 0);
    if (ret < 0)
        return ret;

    ret = av_samples_alloc_array_and_samples((uint8_t ***)&s->orig_chunk, NULL,
                                             s->channels, nb_samples,
                                             AV_SAMPLE_FMT_FLTP, 0);
    if (ret < 0)
        return ret;

    s->buf_allocated_samples = nb_samples;
    return 0;
}

static int push_input_to_fifo(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    MusicDamperContext *s = ctx->priv;
    int expected_pts_valid = 0;
    int64_t expected_next_pts = AV_NOPTS_VALUE;

    if (av_audio_fifo_size(s->fifo) == 0) {
        if (in->pts != AV_NOPTS_VALUE) {
            s->fifo_start_pts = in->pts;
            s->fifo_has_pts = 1;
        } else {
            s->fifo_has_pts = 0;
        }
    } else if (s->fifo_has_pts && in->pts != AV_NOPTS_VALUE) {
        expected_pts_valid = 1;
        expected_next_pts = s->fifo_start_pts +
            av_rescale_q(av_audio_fifo_size(s->fifo),
                         (AVRational){1, inlink->sample_rate}, inlink->time_base);
    }

    if (expected_pts_valid && in->pts != expected_next_pts) {
        av_log(ctx, AV_LOG_WARNING,
               "PTS discontinuity detected while buffering (%" PRId64 " vs expected %" PRId64 ")\n",
               in->pts, expected_next_pts);
    }

    if (av_audio_fifo_write(s->fifo, (void **)in->extended_data, in->nb_samples) < in->nb_samples)
        return AVERROR(ENOMEM);

    return 0;
}

static void copy_chunk(float **dst, float **src, int channels, int n)
{
    int c;
    for (c = 0; c < channels; ++c)
        memcpy(dst[c], src[c], n * sizeof(float));
}

static int output_chunk(AVFilterLink *outlink, int n)
{
    AVFilterContext *ctx = outlink->src;
    MusicDamperContext *s = ctx->priv;
    AVFrame *out = NULL;
    const SherpaOnnxSourceSeparationOutput *sep = NULL;
    int ret = 0;
    int c;
    int use_fallback = 0;

    ret = ensure_chunk_buffers(s, n);
    if (ret < 0)
        return ret;

    if (av_audio_fifo_read(s->fifo, (void **)s->in_chunk, n) < n)
        return AVERROR_BUG;

    copy_chunk(s->orig_chunk, s->in_chunk, s->channels, n);

    if (!s->fallback_passthrough && s->ss) {
        sep = SherpaOnnxOfflineSourceSeparationProcess(
            s->ss,
            (const float *const *)s->in_chunk,
            s->channels,
            n,
            s->sample_rate);

        if (!sep) {
            av_log(ctx, AV_LOG_ERROR,
                   "SherpaOnnxOfflineSourceSeparationProcess() failed. "
                   "Falling back to passthrough for this chunk.\n");
            use_fallback = 1;
        } else if (sep->num_stems < 1 || !sep->stems ||
                   sep->stems[0].num_channels != s->channels ||
                   sep->stems[0].n <= 0) {
            av_log(ctx, AV_LOG_ERROR,
                   "Invalid source-separation output shape. "
                   "Falling back to passthrough for this chunk.\n");
            use_fallback = 1;
        }
    } else {
        use_fallback = 1;
    }

    out = ff_get_audio_buffer(outlink, n);
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto done;
    }

    if (s->fifo_has_pts)
        out->pts = s->fifo_start_pts;

    out->nb_samples = n;

    if (!use_fallback && sep) {
        for (c = 0; c < s->channels; ++c) {
            memcpy(out->extended_data[c], sep->stems[0].samples[c],
                   n * sizeof(float));
        }
    } else {
        for (c = 0; c < s->channels; ++c) {
            memcpy(out->extended_data[c], s->orig_chunk[c], n * sizeof(float));
        }
    }

    update_fifo_pts_after_read(s, outlink, n);

    ret = ff_filter_frame(outlink, out);
    out = NULL;

done:
    if (sep)
        SherpaOnnxDestroySourceSeparationOutput(sep);
    av_frame_free(&out);
    return ret;
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    MusicDamperContext *s = ctx->priv;

    s->channels = inlink->ch_layout.nb_channels;
    s->sample_rate = inlink->sample_rate;

    s->fifo = av_audio_fifo_alloc(AV_SAMPLE_FMT_FLTP, s->channels,
                                  FFMAX(s->chunk_size * 2, 4096));
    if (!s->fifo)
        return AVERROR(ENOMEM);

    return 0;
}

static int init(AVFilterContext *ctx)
{
    MusicDamperContext *s = ctx->priv;
    SherpaOnnxOfflineSourceSeparationConfig config;

    memset(&config, 0, sizeof(config));

    if (!s->model_type)
        s->model_type = av_strdup("uvr");

    if (!s->model_type)
        return AVERROR(ENOMEM);

    if (!av_strcasecmp(s->model_type, "spleeter")) {
        if (!s->spleeter_vocals || !s->spleeter_accompaniment) {
            av_log(ctx, AV_LOG_ERROR,
                   "model_type=spleeter requires both spleeter_vocals and "
                   "spleeter_accompaniment. Falling back to passthrough.\n");
            s->fallback_passthrough = 1;
            return 0;
        }

        config.model.spleeter.vocals = s->spleeter_vocals;
        config.model.spleeter.accompaniment = s->spleeter_accompaniment;
    } else if (!av_strcasecmp(s->model_type, "uvr")) {
        if (!s->uvr_model) {
            av_log(ctx, AV_LOG_ERROR,
                   "model_type=uvr requires uvr_model. "
                   "Falling back to passthrough.\n");
            s->fallback_passthrough = 1;
            return 0;
        }

        config.model.uvr.model = s->uvr_model;
    } else {
        av_log(ctx, AV_LOG_ERROR,
               "Invalid model_type='%s'. Use 'spleeter' or 'uvr'. "
               "Falling back to passthrough.\n",
               s->model_type);
        s->fallback_passthrough = 1;
        return 0;
    }

    config.model.num_threads = s->threads;
    config.model.provider = "cpu";
    config.model.debug = 0;

    s->ss = SherpaOnnxCreateOfflineSourceSeparation(&config);
    if (!s->ss) {
        av_log(ctx, AV_LOG_ERROR,
               "Failed to initialize Sherpa-ONNX source separation engine. "
               "Falling back to passthrough.\n");
        s->fallback_passthrough = 1;
        return 0;
    }

    av_log(ctx, AV_LOG_INFO,
           "musicdamper initialized: model_type=%s threads=%d chunk_size=%d provider=cpu\n",
           s->model_type, s->threads, s->chunk_size);

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    MusicDamperContext *s = ctx->priv;

    if (s->ss)
        SherpaOnnxDestroyOfflineSourceSeparation(s->ss);
    s->ss = NULL;

    av_audio_fifo_free(s->fifo);
    s->fifo = NULL;

    if (s->in_chunk) {
        av_freep(&s->in_chunk[0]);
        av_freep(&s->in_chunk);
    }

    if (s->orig_chunk) {
        av_freep(&s->orig_chunk[0]);
        av_freep(&s->orig_chunk);
    }

    s->buf_allocated_samples = 0;
}

static int activate(AVFilterContext *ctx)
{
    MusicDamperContext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *in = NULL;
    int ret;
    int status;
    int64_t status_pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (s->fallback_passthrough) {
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0)
            return ff_filter_frame(outlink, in);

        if (ff_inlink_acknowledge_status(inlink, &status, &status_pts)) {
            ff_outlink_set_status(outlink, status, status_pts);
            return 0;
        }

        FF_FILTER_FORWARD_WANTED(outlink, inlink);
        return 0;
    }

    while (av_audio_fifo_size(s->fifo) >= s->chunk_size) {
        ret = output_chunk(outlink, s->chunk_size);
        if (ret < 0)
            return ret;
    }

    ret = ff_inlink_consume_frame(inlink, &in);
    if (ret < 0)
        return ret;

    if (ret > 0) {
        ret = push_input_to_fifo(inlink, in);
        av_frame_free(&in);
        if (ret < 0)
            return ret;

        while (av_audio_fifo_size(s->fifo) >= s->chunk_size) {
            ret = output_chunk(outlink, s->chunk_size);
            if (ret < 0)
                return ret;
        }
        return 0;
    }

    if (ff_inlink_acknowledge_status(inlink, &status, &status_pts)) {
        s->input_eof = (status == AVERROR_EOF);

        if (s->input_eof && av_audio_fifo_size(s->fifo) > 0) {
            int tail = av_audio_fifo_size(s->fifo);
            ret = output_chunk(outlink, tail);
            if (ret < 0)
                return ret;
            return 0;
        }

        if (av_audio_fifo_size(s->fifo) == 0)
            ff_outlink_set_status(outlink, status, status_pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);
    return 0;
}

static const AVFilterPad musicdamper_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_AUDIO,
        .config_props = config_input,
    },
};

static const AVFilterPad musicdamper_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_AUDIO,
    },
};

const AVFilter ff_af_musicdamper = {
    .name          = "musicdamper",
    .description   = NULL_IF_CONFIG_SMALL("Sherpa-ONNX source-separation filter that outputs vocals."),
    .priv_size     = sizeof(MusicDamperContext),
    .priv_class    = &musicdamper_class,
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    FILTER_INPUTS(musicdamper_inputs),
    FILTER_OUTPUTS(musicdamper_outputs),
    FILTER_QUERY_FUNC2(query_formats),
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};
