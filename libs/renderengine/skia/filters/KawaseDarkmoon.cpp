/*
 * Copyright 2026 KamiKaonashi
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define ATRACE_TAG ATRACE_TAG_GRAPHICS

#include "KawaseDarkmoon.h"

#include <algorithm>
#include <cmath>

#include <SkAlphaType.h>
#include <SkBlendMode.h>
#include <SkCanvas.h>
#include <SkColorSpace.h>
#include <SkColorType.h>
#include <SkImageInfo.h>
#include <SkPaint.h>
#include <SkRect.h>
#include <SkRuntimeEffect.h>
#include <SkSamplingOptions.h>
#include <SkShader.h>
#include <SkSize.h>
#include <SkString.h>
#include <SkSurface.h>
#include <SkTileMode.h>
#include <common/trace.h>
#include <log/log.h>

#include "RuntimeEffectManager.h"

namespace android {
namespace renderengine {
namespace skia {

namespace {

constexpr float kBlurSigmaScale = 0.57735f;

constexpr float kDiagonalKernelCompensation = 0.93541435f;

constexpr float kClampInset = 0.5001f;

const SkSamplingOptions kLinearSampling(SkFilterMode::kLinear, SkMipmapMode::kNone);

int levelDimension(int fullResDimension, int level) {
    const float scale = KawaseDarkmoon::kInverseInputScale * static_cast<float>(1 << level);
    return std::max(1, static_cast<int>(static_cast<float>(fullResDimension) / scale));
}

SkISize levelSize(SkISize fullRes, int level) {
    return SkISize::Make(levelDimension(fullRes.width(), level),
                         levelDimension(fullRes.height(), level));
}

SkRect insetForClamp(const SkRect& rect) {
    SkRect inset = rect.makeInset(kClampInset, kClampInset);
    if (inset.width() < 0) {
        inset.fLeft = inset.fRight = rect.centerX();
    }
    if (inset.height() < 0) {
        inset.fTop = inset.fBottom = rect.centerY();
    }
    return inset;
}

bool isOversized(const sk_sp<SkSurface>& surface, SkISize content) {
    return surface->width() > content.width() || surface->height() > content.height();
}

bool isCompatible(const SkImageInfo& a, const SkImageInfo& b) {
    return a.colorType() == b.colorType() && a.alphaType() == b.alphaType() &&
            SkColorSpace::Equals(a.colorSpace(), b.colorSpace());
}

} // namespace

const SkString kEffectSource_KawaseDarkmoon_QuarterResDownSampleBlurEffect(R"(
    uniform shader child;

    const float2 STEP_0 = float2( 0.25,  0.25);
    const float2 STEP_1 = float2( 0.25, -0.25);
    const float2 STEP_2 = float2(-0.25, -0.25);
    const float2 STEP_3 = float2(-0.25,  0.25);

    half4 main(float2 xy) {
        half3 c = child.eval(xy + STEP_0).rgb;
        c += child.eval(xy + STEP_1).rgb;
        c += child.eval(xy + STEP_2).rgb;
        c += child.eval(xy + STEP_3).rgb;

        return half4(c * 0.25, 1.0);
    }
)");

const SkString kEffectSource_KawaseDarkmoon_HalfResDownSampleBlurEffect(R"(
    uniform shader child;

    const float2 STEP_0 = float2( 0.5,  0.5);
    const float2 STEP_1 = float2( 0.5, -0.5);
    const float2 STEP_2 = float2(-0.5, -0.5);
    const float2 STEP_3 = float2(-0.5,  0.5);

    half4 main(float2 xy) {
        half3 c = child.eval(xy + STEP_0).rgb;
        c += child.eval(xy + STEP_1).rgb;
        c += child.eval(xy + STEP_2).rgb;
        c += child.eval(xy + STEP_3).rgb;

        return half4(c * 0.25, 1.0);
    }
)");

const SkString kEffectSource_KawaseDarkmoon_UpSampleBlurEffect(R"(
    uniform shader child;
    uniform float in_blurOffset;
    uniform float in_crossFade;
    uniform float in_weightedCrossFade;

    half4 main(float2 xy) {
        half3 c = child.eval(xy).rgb * 4.0;
        c += child.eval(xy + float2( in_blurOffset,  in_blurOffset)).rgb;
        c += child.eval(xy + float2( in_blurOffset, -in_blurOffset)).rgb;
        c += child.eval(xy + float2(-in_blurOffset, -in_blurOffset)).rgb;
        c += child.eval(xy + float2(-in_blurOffset,  in_blurOffset)).rgb;

        return half4(c * in_weightedCrossFade, in_crossFade);
    }
)");

KawaseDarkmoon::KawaseDarkmoon(RuntimeEffectManager& effectManager) : BlurFilter(effectManager) {
    mQuarterResDownSampleBlurEffect =
            effectManager.mKnownEffects[kKawaseDarkmoon_QuarterResDownSampleBlurEffect];
    mHalfResDownSampleBlurEffect =
            effectManager.mKnownEffects[kKawaseDarkmoon_HalfResDownSampleBlurEffect];
    mUpSampleBlurEffect = effectManager.mKnownEffects[kKawaseDarkmoon_UpSampleBlurEffect];
}

sk_sp<SkShader> KawaseDarkmoon::makeBlurShader(const sk_sp<SkImage>& blurredImage,
                                               const SkMatrix& blurMatrix,
                                               const SkRect& blurRect) const {
    SkIRect targetBlurRect;
    blurRect.roundIn(&targetBlurRect);
    const SkRect content = SkRect::Make(levelSize(targetBlurRect.size(), 0));
    return SkShaders::CoordClamp(blurredImage->makeShader(SkTileMode::kClamp, SkTileMode::kClamp,
                                                          kLinearSampling),
                                 insetForClamp(content))
            ->makeWithLocalMatrix(blurMatrix);
}

bool KawaseDarkmoon::ensureSurface(SurfaceCache& cache, int index, SkiaGpuContext* context,
                                   const SkImageInfo& info, SkISize required,
                                   SkISize preferred) const {
    sk_sp<SkSurface>& surface = cache.surfaces[index];
    if (surface) {
        const SkImageInfo& current = surface->imageInfo();
        const bool compatible = isCompatible(current, info);
        if (compatible && current.width() >= required.width() &&
            current.height() >= required.height()) {
            return true;
        }
        if (compatible) {
            preferred.fWidth = std::max(preferred.width(), current.width());
            preferred.fHeight = std::max(preferred.height(), current.height());
        }
        surface.reset();
    }

    surface = context->createRenderTarget(info.makeWH(preferred.width(), preferred.height()));
    if (!surface) {
        context->purgeUnlockedScratchResources();
        surface = context->createRenderTarget(info.makeWH(preferred.width(), preferred.height()));
        if (!surface && preferred != required) {
            surface = context->createRenderTarget(info.makeWH(required.width(), required.height()));
        }
    }
    return surface != nullptr;
}

void KawaseDarkmoon::blurInto(const sk_sp<SkSurface>& drawSurface, SkISize drawContent,
                              sk_sp<SkShader> input, const SkRect* inputClampRect,
                              const SkMatrix& inputToDest, float radius, float alpha,
                              const sk_sp<SkRuntimeEffect>& blurEffect) const {
    if (inputClampRect) {
        input = SkShaders::CoordClamp(std::move(input), insetForClamp(*inputClampRect));
    }
    input = input->makeWithLocalMatrix(inputToDest);

    SkPaint paint;
    if (blurEffect == mUpSampleBlurEffect && radius == 0) {
        paint.setShader(std::move(input));
        paint.setAlphaf(alpha);
    } else {
        SkRuntimeShaderBuilder blurBuilder(blurEffect);
        blurBuilder.child("child") = std::move(input);
        if (blurEffect == mUpSampleBlurEffect) {
            blurBuilder.uniform("in_blurOffset") = radius;
            blurBuilder.uniform("in_crossFade") = alpha;
            blurBuilder.uniform("in_weightedCrossFade") = alpha * 0.125f;
        }
        paint.setShader(blurBuilder.makeShader(nullptr));
    }
    paint.setBlendMode(alpha == 1.0f ? SkBlendMode::kSrc : SkBlendMode::kSrcOver);

    SkCanvas* canvas = drawSurface->getCanvas();
    SkAutoCanvasRestore autoRestore(canvas, true);
    const bool oversized = isOversized(drawSurface, drawContent);
    if (alpha == 1.0f && !oversized) {
        // The kSrc draw overwrites every pixel, so the old contents are not needed. Discarding first
        // avoids loading them, which costs DRAM bandwidth on tile-based GPUs. Not done when the draw is
        // clipped, since pixels outside the clip keep their contents.
        canvas->discard();
    }
    if (oversized) {
        canvas->clipRect(SkRect::Make(drawContent));
    }
    canvas->drawPaint(paint);
}

// The downsampled levels of one blur input. They depend only on the input and the rectangle, never on
// the blur radius, so the blurs of one layer can share them. Each radius then runs only its upsample
// passes. The levels live in mSharedDownCache, which holds one prepared input at a time. The layer blur
// block prepares one, uses it for all of its blurs, and drops it before the next layer.
class KawaseDarkmoon::PreparedInput : public BlurFilter::PreparedBlurInput {
public:
    PreparedInput(const KawaseDarkmoon* filter, SkiaGpuContext* context,
                  const DisplaySettings& display, const sk_sp<SkImage>& input,
                  const SkRect& blurRect, uint64_t generation)
          : mFilter(filter), mContext(context), mInput(input), mGeneration(generation) {
        SkIRect targetBlurRect;
        blurRect.roundIn(&targetBlurRect);
        mBlurRect = blurRect;
        mInputClamp = SkRect::Make(targetBlurRect);
        mFullRes = targetBlurRect.size();
        mDisplaySize = SkISize::Make(display.physicalDisplay.width(), display.physicalDisplay.height());
        mHintFromDisplay = !mDisplaySize.isEmpty();
        mBaseInfo = input->imageInfo();
        for (int i = 0; i < kMaxSurfaces; i++) {
            mContentSizes[i] = levelSize(mFullRes, i);
        }
    }

    SkiaGpuContext* context() const { return mContext; }
    const sk_sp<SkImage>& input() const { return mInput; }
    const SkImageInfo& baseInfo() const { return mBaseInfo; }
    SkISize contentSize(int i) const { return mContentSizes[i]; }
    const sk_sp<SkSurface>& level(int i) const { return mFilter->mSharedDownCache.surfaces[i]; }

    SkISize preferredSize(int i) const {
        SkISize preferred = mContentSizes[i];
        if (mHintFromDisplay) {
            const SkISize displayLevel = levelSize(mDisplaySize, i);
            preferred.fWidth = std::max(preferred.width(), displayLevel.width());
            preferred.fHeight = std::max(preferred.height(), displayLevel.height());
        }
        return preferred;
    }

    // Makes levels 0..depth available. Returns false if a target could not be created, or if another
    // layer has taken the shared levels since this input was prepared.
    bool ensureLevels(int depth) {
        if (mFilter->mChainGeneration != mGeneration) {
            return false;
        }
        for (; mComputedLevels <= depth; mComputedLevels++) {
            const int i = mComputedLevels;
            if (!mFilter->ensureSurface(mFilter->mSharedDownCache, i, mContext, mBaseInfo,
                                        mContentSizes[i], preferredSize(i))) {
                return false;
            }
            if (i == 0) {
                // Skia's local matrix maps shader space -> destination, hence translate-then-scale.
                SkMatrix blurMatrix = SkMatrix::Translate(-mBlurRect.fLeft, -mBlurRect.fTop);
                blurMatrix.postScale(kInputScale, kInputScale);
                mFilter->blurInto(mFilter->mSharedDownCache.surfaces[0], mContentSizes[0],
                                  mInput->makeShader(SkTileMode::kClamp, SkTileMode::kClamp,
                                                     kLinearSampling),
                                  &mInputClamp, blurMatrix, 0, 1.0f,
                                  mFilter->mQuarterResDownSampleBlurEffect);
            } else {
                const SkISize src = mContentSizes[i - 1];
                const SkISize dst = mContentSizes[i];
                const SkMatrix toDest =
                        SkMatrix::Scale(static_cast<float>(dst.width()) / src.width(),
                                        static_cast<float>(dst.height()) / src.height());
                const SkRect srcClamp = SkRect::Make(src);
                const sk_sp<SkSurface>& prev = mFilter->mSharedDownCache.surfaces[i - 1];
                const bool clampSrc = isOversized(prev, src);
                mFilter->blurInto(mFilter->mSharedDownCache.surfaces[i], dst,
                                  prev->makeTemporaryImage()->makeShader(
                                          SkTileMode::kClamp, SkTileMode::kClamp, kLinearSampling),
                                  clampSrc ? &srcClamp : nullptr, toDest, 0, 1.0f,
                                  mFilter->mHalfResDownSampleBlurEffect);
            }
        }
        return true;
    }

private:
    const KawaseDarkmoon* mFilter;
    SkiaGpuContext* mContext;
    sk_sp<SkImage> mInput;
    uint64_t mGeneration;
    SkRect mBlurRect;
    SkRect mInputClamp;
    SkISize mFullRes = SkISize::MakeEmpty();
    SkISize mDisplaySize = SkISize::MakeEmpty();
    bool mHintFromDisplay = false;
    SkImageInfo mBaseInfo;
    SkISize mContentSizes[kMaxSurfaces];
    int mComputedLevels = 0;
};

std::unique_ptr<BlurFilter::PreparedBlurInput> KawaseDarkmoon::prepareBlurInput(
        SkiaGpuContext* context, const DisplaySettings& display, const sk_sp<SkImage>& input,
        const SkRect& blurRect) const {
    // Protected content takes the transient path, so it cannot share levels.
    if (!context || !input || context->supportsProtectedContent()) {
        return nullptr;
    }
    if (mSharedDownCache.context != context) {
        mSharedDownCache.reset();
        mSharedDownCache.context = context;
    }
    if (mSharedUpCache.context != context) {
        mSharedUpCache.reset();
        mSharedUpCache.context = context;
    }
    const uint64_t generation = ++mChainGeneration;
    return std::make_unique<PreparedInput>(this, context, display, input, blurRect, generation);
}

sk_sp<SkImage> KawaseDarkmoon::generateFromPreparedInput(BlurFilter::PreparedBlurInput& prepared,
                                                         uint32_t blurRadius) const {
    SFTRACE_CALL();
    PreparedInput& p = static_cast<PreparedInput&>(prepared);
    if (blurRadius == 0) {
        return p.input();
    }

    // Same radius and pass arithmetic as generateTemporaryImage, so the result is identical.
    const float radius = blurRadius * kBlurSigmaScale;
    const float filterDepth = std::min(kMaxSurfaces - 1.0f, radius * kInputScale / 2.5f);
    const int filterPasses = std::min(kMaxSurfaces - 1, static_cast<int>(std::ceil(filterDepth)));
    if (!p.ensureLevels(filterPasses)) {
        return nullptr;
    }
    if (filterPasses == 0) {
        return p.level(0)->makeTemporaryImage();
    }
    float sumSquaredR = 0;
    float sumSquaredStep = 0;
    for (int i = 0; i < filterPasses; i++) {
        const float alpha = std::min(1.0f, filterDepth - i);
        sumSquaredR += powf(powf(2.0f, i - 1) * alpha * static_cast<float>(M_SQRT2), 2.0f);
        sumSquaredStep += powf(powf(2.0f, i) * alpha, 2.0f);
    }
    const float step = kDiagonalKernelCompensation *
            sqrtf(std::max(0.0f, powf(radius * kInputScale, 2.0f) - sumSquaredR) /
                  (sumSquaredStep == 0 ? 1.0f : sumSquaredStep));


    // Upsample targets have their own cache, so they never overwrite a shared level.
    for (int i = 0; i < filterPasses; i++) {
        if (!ensureSurface(mSharedUpCache, i, p.context(), p.baseInfo(), p.contentSize(i),
                           p.preferredSize(i))) {
            return nullptr;
        }
    }
    const sk_sp<SkSurface>* up = mSharedUpCache.surfaces;
    for (int i = filterPasses - 1; i >= 0; i--) {
        const SkISize src = p.contentSize(i + 1);
        const SkISize dst = p.contentSize(i);
        const SkMatrix toDest = SkMatrix::Scale(static_cast<float>(dst.width()) / src.width(),
                                                static_cast<float>(dst.height()) / src.height());
        const SkRect srcClamp = SkRect::Make(src);
        const sk_sp<SkSurface>& srcSurface = (i + 1 == filterPasses) ? p.level(i + 1) : up[i + 1];
        const bool clampSrc = isOversized(srcSurface, src);
        const float alpha = std::min(1.0f, filterDepth - i);
        if (alpha < 1.0f) {
            // A partial pass blends with what its target already holds, which must be level i, as in
            // generateTemporaryImage. The shared upsample target holds an earlier blur's result, so copy
            // level i into it first.
            SkPaint copy;
            copy.setBlendMode(SkBlendMode::kSrc);
            up[i]->getCanvas()->drawImage(p.level(i)->makeTemporaryImage(), 0, 0,
                                          SkSamplingOptions(), &copy);
        }
        blurInto(up[i], dst,
                 srcSurface->makeTemporaryImage()->makeShader(SkTileMode::kClamp,
                                                              SkTileMode::kClamp, kLinearSampling),
                 clampSrc ? &srcClamp : nullptr, toDest, step, alpha, mUpSampleBlurEffect);
    }
    return up[0]->makeTemporaryImage();
}

sk_sp<SkImage> KawaseDarkmoon::generateTemporaryImage(SkiaGpuContext* context,
                                                      const DisplaySettings& display,
                                                      const uint32_t blurRadius,
                                                      const sk_sp<SkImage> input,
                                                      const SkRect& blurRect) const {
    SFTRACE_CALL();
    LOG_ALWAYS_FATAL_IF(context == nullptr, "%s: Needs GPU context", __func__);
    LOG_ALWAYS_FATAL_IF(input == nullptr, "%s: Invalid input image", __func__);

    if (blurRadius == 0) {
        return input;
    }

    const float radius = blurRadius * kBlurSigmaScale;

    const float filterDepth = std::min(kMaxSurfaces - 1.0f, radius * kInputScale / 2.5f);
    const int filterPasses = std::min(kMaxSurfaces - 1, static_cast<int>(std::ceil(filterDepth)));

    SkIRect targetBlurRect;
    blurRect.roundIn(&targetBlurRect);
    const SkISize fullRes = targetBlurRect.size();

    const bool isProtected = context->supportsProtectedContent();
    SurfaceCache transientCache;
    SurfaceCache& cache = isProtected ? transientCache : mUnprotectedCache;
    if (cache.context != context) {
        cache.reset();
        cache.context = context;
    }

    const SkISize displaySize =
            SkISize::Make(display.physicalDisplay.width(), display.physicalDisplay.height());
    const bool hintFromDisplay = !isProtected && !displaySize.isEmpty();

    const SkImageInfo baseInfo = input->imageInfo();

    SkISize contentSizes[kMaxSurfaces];
    for (int i = 0; i <= filterPasses; i++) {
        contentSizes[i] = levelSize(fullRes, i);
        SkISize preferred = contentSizes[i];
        if (hintFromDisplay) {
            const SkISize displayLevel = levelSize(displaySize, i);
            preferred.fWidth = std::max(preferred.width(), displayLevel.width());
            preferred.fHeight = std::max(preferred.height(), displayLevel.height());
        }
        const bool allocated =
                ensureSurface(cache, i, context, baseInfo, contentSizes[i], preferred);
        LOG_ALWAYS_FATAL_IF(!allocated, "%s: Failed to create surface %d (%dx%d) for blurring!",
                            __func__, i, contentSizes[i].width(), contentSizes[i].height());
    }
    const sk_sp<SkSurface>* surfaces = cache.surfaces;

    float sumSquaredR = 0;
    float sumSquaredStep = 0;
    for (int i = 0; i < filterPasses; i++) {
        const float alpha = std::min(1.0f, filterDepth - i);
        sumSquaredR += powf(powf(2.0f, i - 1) * alpha * static_cast<float>(M_SQRT2), 2.0f);
        sumSquaredStep += powf(powf(2.0f, i) * alpha, 2.0f);
    }
    const float step = kDiagonalKernelCompensation *
            sqrtf(std::max(0.0f, powf(radius * kInputScale, 2.0f) - sumSquaredR) /
                  (sumSquaredStep == 0 ? 1.0f : sumSquaredStep));

    // First pass: down-sample from the input at BlurFilter::kInputScale.
    {
        // Skia's local matrix maps shader space -> destination, hence translate-then-scale.
        SkMatrix blurMatrix = SkMatrix::Translate(-blurRect.fLeft, -blurRect.fTop);
        blurMatrix.postScale(kInputScale, kInputScale);
        const SkRect inputClamp = SkRect::Make(targetBlurRect);
        blurInto(surfaces[0], contentSizes[0],
                 input->makeShader(SkTileMode::kClamp, SkTileMode::kClamp, kLinearSampling),
                 &inputClamp, blurMatrix,
                 0, // unused: the down-sample effect hardcodes its offsets
                 1.0f, mQuarterResDownSampleBlurEffect);
    }

    // Remaining 2x down-samples.
    for (int i = 0; i < filterPasses; i++) {
        const SkISize src = contentSizes[i];
        const SkISize dst = contentSizes[i + 1];
        const SkMatrix toDest = SkMatrix::Scale(static_cast<float>(dst.width()) / src.width(),
                                                static_cast<float>(dst.height()) / src.height());
        const SkRect srcClamp = SkRect::Make(src);
        const bool clampSrc = isOversized(surfaces[i], src);
        blurInto(surfaces[i + 1], dst,
                 surfaces[i]->makeTemporaryImage()->makeShader(SkTileMode::kClamp,
                                                               SkTileMode::kClamp, kLinearSampling),
                 clampSrc ? &srcClamp : nullptr, toDest,
                 0, // unused: the down-sample effect hardcodes its offsets
                 1.0f, mHalfResDownSampleBlurEffect);
    }

    // Blur + up-sample back to the initial down-sampled resolution.
    for (int i = filterPasses - 1; i >= 0; i--) {
        const SkISize src = contentSizes[i + 1];
        const SkISize dst = contentSizes[i];
        const SkMatrix toDest = SkMatrix::Scale(static_cast<float>(dst.width()) / src.width(),
                                                static_cast<float>(dst.height()) / src.height());
        const SkRect srcClamp = SkRect::Make(src);
        const bool clampSrc = isOversized(surfaces[i + 1], src);
        blurInto(surfaces[i], dst,
                 surfaces[i + 1]->makeTemporaryImage()->makeShader(SkTileMode::kClamp,
                                                                   SkTileMode::kClamp,
                                                                   kLinearSampling),
                 clampSrc ? &srcClamp : nullptr, toDest, step, std::min(1.0f, filterDepth - i),
                 mUpSampleBlurEffect);
    }

    // The result is at kInputScale relative to blurRect and anchored at the surface's origin,
    // which is what BlurFilter::drawBlurRegion() expects (it maps with a fixed inverse scale).
    return surfaces[0]->makeTemporaryImage();
}

} // namespace skia
} // namespace renderengine
} // namespace android
