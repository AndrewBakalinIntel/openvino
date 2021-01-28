// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/merge_gather_gather_elements.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <vpu/ngraph/operations/exp_gather_elements.hpp>
#include "vpu/ngraph/utilities.hpp"

#include <ngraph/pass/visualize_tree.hpp>

NGRAPH_RTTI_DEFINITION(vpu::MergeGatherGatherElements, "MergeGatherGatherElements", 0);

namespace vpu {

bool MergeGatherGatherElements::run_on_function(std::shared_ptr<ngraph::Function> f) {
    bool graphChanged = false;

    ngraph::pass::VisualizeTree("/home/abakalin/work/temp/gge.svg").run_on_function(f);

    const auto gatherData = ngraph::pattern::any_input();
    const auto gatherIndices = ngraph::pattern::any_input();

    const auto gather = ngraph::pattern::wrap_type<ngraph::opset6::Gather>({gatherData, gatherIndices, ngraph::pattern::any_input()});

    const auto squeezeAxis = ngraph::pattern::any_input();
    const auto squeeze = ngraph::pattern::wrap_type<ngraph::opset6::Squeeze>({gather, squeezeAxis});
    const auto transposePerm = ngraph::pattern::any_input();
    const auto transpose = ngraph::pattern::wrap_type<ngraph::opset6::Transpose>({squeeze, transposePerm});

    const auto m = std::make_shared<ngraph::pattern::Matcher>(transpose, "GatherSqueezeTransposeMatcher");
    for (const auto& node : f->get_ordered_ops()) {
        if (m->match(node)) {
            auto& patternMap = m->get_pattern_value_map();

            const auto& m_transpose = patternMap.at(transpose);
            std::vector<ngraph::Node*> shapeOfs;
            std::vector<ngraph::Node*> gatherElements;
            for (auto& transposeConsumer : m_transpose.get_target_inputs()) {
                if (transposeConsumer.get_node()->get_type_info() == ngraph::opset6::ShapeOf::type_info) {
                    shapeOfs.push_back(transposeConsumer.get_node());
                } else if (transposeConsumer.get_node()->get_type_info() == ngraph::opset6::GatherElements::type_info) {
                    gatherElements.push_back(ngraph::as_type<ngraph::opset6::GatherElements>(transposeConsumer.get_node()));
                } else {
                    continue;
                }
            }

            const auto& m_transposePerm = patternMap.at(transposePerm);
            const auto& m_squeezeAxis = patternMap.at(squeezeAxis);
            const auto& m_gatherIndices = patternMap.at(gatherIndices);
            const auto& m_gather = patternMap.at(gather);
            const auto& m_squeeze = patternMap.at(squeeze);
            for (const auto& gatherElement : gatherElements) {
                const auto transposeIndices = std::make_shared<ngraph::opset6::Transpose>(gatherElement->input_value(1), m_transposePerm);
                const auto unsqueeze = std::make_shared<ngraph::opset6::Unsqueeze>(transposeIndices, m_squeezeAxis);
                const auto expGatherElements = std::make_shared<ngraph::vpu::op::ExpGatherElements>(
                    gatherElement->input_value(0),
                    unsqueeze,
                    m_gatherIndices,
                    ngraph::as_type<ngraph::opset6::GatherElements>(gatherElement)->get_axis(),
                    ngraph::as_type<ngraph::opset6::Gather>(m_gather.get_node())->get_axis());
                const auto squeezeData = m_squeeze.get_node()->clone_with_new_inputs({expGatherElements, m_squeezeAxis});
                const auto transposeData = m_transpose.get_node()->clone_with_new_inputs({squeezeData, m_transposePerm});
                gatherElement->output(0).replace(transposeData);
            }

            const auto& m_gatherData = patternMap.at(gatherData);
            const auto gatherDataShape = std::make_shared<ngraph::opset6::ShapeOf>(m_gatherData, ngraph::element::i32);
            const auto gatherIndicesShape = std::make_shared<ngraph::opset6::ShapeOf>(m_gatherIndices, ngraph::element::i32);

            ngraph::OutputVector outputDims;
            const auto axis = ngraph::as_type<ngraph::opset6::Gather>(m_gather.get_node())->get_axis();
            outputDims.push_back(gatherShapeElements(gatherDataShape, 0, axis));
            outputDims.push_back(gatherIndicesShape);
            outputDims.push_back(gatherShapeElements(gatherDataShape, axis + 1, ngraph::shape_size(gatherDataShape->get_shape()) - axis - 1));
            const auto outputShape = std::make_shared<ngraph::opset6::Concat>(outputDims, 0);

            for (const auto& shapeOf : shapeOfs) {
                shapeOf->output(0).replace(outputShape);
            }

            graphChanged = true;
        }
    }

    ngraph::pass::VisualizeTree("/home/abakalin/work/temp/gge_transformed.svg").run_on_function(f);

    return graphChanged;
}

}  // namespace vpu
