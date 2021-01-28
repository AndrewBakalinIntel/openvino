// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph { namespace vpu { namespace op {

class ExpGatherElements : public ngraph::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    explicit ExpGatherElements(const Output<Node>& data,
                               const Output<Node>& indices,
                               const Output<Node>& lookupIndices,
                               const int64_t axis,
                               const int64_t lookupAxis);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node>
    clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_axis() const { return m_axis; }
    int64_t get_lookup_axis() const { return m_lookup_axis; }
private:
    int64_t m_axis;
    int64_t m_lookup_axis;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
