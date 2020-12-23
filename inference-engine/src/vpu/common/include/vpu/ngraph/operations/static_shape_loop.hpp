// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset5.hpp>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeLoop : public ngraph::opset5::Loop {
public:
    NGRAPH_RTTI_DECLARATION;

    explicit StaticShapeLoop(const Loop& loop);
    void validate_and_infer_types() override;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
