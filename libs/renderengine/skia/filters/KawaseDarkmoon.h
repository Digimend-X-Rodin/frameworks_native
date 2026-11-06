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

#pragma once

#include <SkImage.h>
#include <SkImageInfo.h>
#include <SkMatrix.h>
#include <SkRect.h>
#include <SkRuntimeEffect.h>
#include <SkShader.h>
#include <SkSize.h>
#include <SkSurface.h>
#include "BlurFilter.h"
#include "RuntimeEffectManager.h"

namespace android {
namespace renderengine {
namespace skia {

/**
 * KawaseDarkmoon - a lightweight dual-filter Kawase blur aimed at low-end GPUs.
 *
 * Compared to KawaseBlurDualFilterV2 it:
 *  - uses 4 taps instead of 5 in subsequent down-sample passes and 5 instead of 8 in
 *    up-sample passes (the initial down-sample uses the same 4 taps), and
 *  - reuses unprotected SkSurfaces across frames, growing to fit the display and blur region
 *    while the GPU context and image format stay compatible. This is independent of V2's
 *    optional hardware-buffer preallocation.
 *
 * Intermediate surfaces may be larger than the region being blurred. All reads are
 * coordinate-clamped to the valid content, including the final composition shader, so stale
 * pixels from earlier (larger) blurs never leak into the result.
 *
 * Protected (DRM) content is never cached: protected memory is scarce, so those blurs use
 * exactly-sized transient surfaces.
 */
class KawaseDarkmoon : public BlurFilter {
public:
    static constexpr int kMaxSurfaces = 4;
    static constexpr float kInverseInputScale = 1.0f / kInputScale;

    explicit KawaseDarkmoon(RuntimeEffectManager& effectManager);
    virtual ~KawaseDarkmoon() {}

    sk_sp<SkImage> generateTemporaryImage(SkiaGpuContext* context, const DisplaySettings& display,
                                          const uint32_t radius, const sk_sp<SkImage> blurInput,
                                          const SkRect& blurRect) const override;

private:
    sk_sp<SkShader> makeBlurShader(const sk_sp<SkImage>& blurredImage, const SkMatrix& blurMatrix,
                                   const SkRect& blurRect) const override;

    struct SurfaceCache {
        SkiaGpuContext* context = nullptr;
        sk_sp<SkSurface> surfaces[kMaxSurfaces];

        void reset() {
            context = nullptr;
            for (auto& surface : surfaces) {
                surface.reset();
            }
        }
    };

    sk_sp<SkRuntimeEffect> mQuarterResDownSampleBlurEffect;
    sk_sp<SkRuntimeEffect> mHalfResDownSampleBlurEffect;
    sk_sp<SkRuntimeEffect> mUpSampleBlurEffect;

    mutable SurfaceCache mUnprotectedCache;

    bool ensureSurface(SurfaceCache& cache, int index, SkiaGpuContext* context,
                       const SkImageInfo& info, SkISize required, SkISize preferred) const;

    void blurInto(const sk_sp<SkSurface>& drawSurface, SkISize drawContent, sk_sp<SkShader> input,
                  const SkRect* inputClampRect, const SkMatrix& inputToDest, float radius,
                  float alpha, const sk_sp<SkRuntimeEffect>& blurEffect) const;
};

} // namespace skia
} // namespace renderengine
} // namespace android
