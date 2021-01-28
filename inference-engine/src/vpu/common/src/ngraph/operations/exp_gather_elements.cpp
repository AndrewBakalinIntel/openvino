// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/exp_gather_elements.hpp"

namespace ngraph { namespace vpu { namespace op {

NGRAPH_RTTI_DEFINITION(ExpGatherElements, "ExpGatherElements", 0);

ExpGatherElements::ExpGatherElements(const Output<Node>& data,
                                     const Output<Node>& indices,
                                     const Output<Node>& lookupIndices,
                                     const int64_t axis,
                                     const int64_t lookupAxis)
    : ngraph::op::Op({data, indices, lookupIndices})
    , m_axis(axis)
    , m_lookup_axis(lookupAxis) {
    constructor_validate_and_infer_types();
}

void ExpGatherElements::validate_and_infer_types() {
    const auto& indicesShape = get_input_partial_shape(1);
    const auto& dataType = get_input_element_type(0);

    set_output_type(0, dataType, indicesShape);
}

bool ExpGatherElements::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("lookup_axis", m_lookup_axis);
    return true;
}

std::shared_ptr<Node> ExpGatherElements::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ExpGatherElements>(new_args.at(0), new_args.at(1), new_args.at(2), get_axis(), m_lookup_axis);
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
