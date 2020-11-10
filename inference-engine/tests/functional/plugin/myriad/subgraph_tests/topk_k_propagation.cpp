// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/function.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/operations/static_shape_topk.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_topk.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>

namespace {

class DynamicToStaticTopKPropagationConcatBased : public CommonTestUtils::TestsCommon,
                                                  public testing::WithParamInterface<int64_t> {
public:
    void SetUp() override {
        const auto& k = GetParam();
        ngraph::Shape inputShape{upperBoundK};

        const auto data = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, inputShape);
        const auto dataShape = ngraph::opset5::Constant::create(ngraph::element::i32, ngraph::Shape{inputShape.size()}, inputShape);
        const auto resultK = ngraph::opset5::Constant::create(ngraph::element::i32, {1}, {k});

        const auto concat = std::make_shared<ngraph::opset5::Concat>(ngraph::OutputVector{dataShape, resultK}, 0);

        const auto reduceMin = std::make_shared<ngraph::opset5::ReduceMin>(concat, ngraph::opset5::Constant::create(ngraph::element::i32, {1}, {0}), false);
        const auto builtSubgraph = buildSubgraph(reduceMin);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dataShape);
        const auto topK = std::make_shared<ngraph::opset5::TopK>(dsr, builtSubgraph, 0, "max", "value");

        ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(topK->output(0)),
                                     std::make_shared<ngraph::opset5::Result>(topK->output(1))};
        const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{data}, "TopKPropagationOfK");

        const auto transformations = vpu::Transformations{{topK->type_info, vpu::dynamicToStaticShapeTopK}};
        ASSERT_NO_THROW(vpu::DynamicToStaticShape(transformations).run_on_function(function));
        validate(*function);
    }

protected:
    void validate(const ngraph::Function& function) const {
        const auto& k = GetParam();

        ngraph::ResultVector processedResults;
        ASSERT_NO_THROW(processedResults = function.get_results());
        EXPECT_EQ(processedResults.size(), 2);

        const auto topKOutPartialShape = processedResults[0]->get_input_partial_shape(0);
        EXPECT_TRUE(topKOutPartialShape.is_static());

        const auto topKOutShape = topKOutPartialShape.get_shape();
        EXPECT_EQ(topKOutShape.size(), 1);
        EXPECT_EQ(topKOutShape[0], k);
    }

    virtual std::shared_ptr<ngraph::Node> buildSubgraph(std::shared_ptr<ngraph::Node> node) const {
        return node;
    }

    static constexpr int64_t upperBoundK = 1000;
};

const std::vector<int64_t> kVec = {0, 10, 100, 200, 500};

TEST_P(DynamicToStaticTopKPropagationConcatBased, KPropagation) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticTopKPropagationConcatBased, ::testing::ValuesIn(kVec));

class DynamicToStaticTopKPropagationReshape : public DynamicToStaticTopKPropagationConcatBased {
protected:
    std::shared_ptr<ngraph::Node> buildSubgraph(std::shared_ptr<ngraph::Node> node) const override {
        return std::make_shared<ngraph::opset5::Reshape>(node, ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}), false);
    }
};

TEST_P(DynamicToStaticTopKPropagationReshape, KPropagation) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticTopKPropagationReshape, ::testing::ValuesIn(kVec));

class DynamicToStaticTopKPropagationSqueezeUnsqueeze : public DynamicToStaticTopKPropagationConcatBased {
protected:
    std::shared_ptr<ngraph::Node> buildSubgraph(std::shared_ptr<ngraph::Node> node) const override {
        const auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(node, ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        return std::make_shared<ngraph::opset5::Squeeze>(unsqueeze, ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
    }
};

TEST_P(DynamicToStaticTopKPropagationSqueezeUnsqueeze, KPropagation) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticTopKPropagationSqueezeUnsqueeze, ::testing::ValuesIn(kVec));

class DynamicToStaticTopKPropagationConvert : public DynamicToStaticTopKPropagationConcatBased {
protected:
    std::shared_ptr<ngraph::Node> buildSubgraph(std::shared_ptr<ngraph::Node> node) const override {
        const auto convert = std::make_shared<ngraph::opset5::Convert>(node, ngraph::element::i32);
        return std::make_shared<ngraph::opset5::Convert>(convert, ngraph::element::i64);
    }
};

TEST_P(DynamicToStaticTopKPropagationConvert, KPropagation) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticTopKPropagationConvert, ::testing::ValuesIn(kVec));

class DynamicToStaticTopKPropagationShapeOfBased : public DynamicToStaticTopKPropagationConcatBased {
public:
    void SetUp() override {
        const auto& k = GetParam();

        const auto upperBoundData = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{upperBoundK});
        const auto realData = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ngraph::Shape{static_cast<size_t>(k)});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
            upperBoundData,
            ngraph::opset5::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {k}));

        const auto shapeOf = std::make_shared<ngraph::opset5::ShapeOf>(realData);
        const auto builtSubgraph = buildSubgraph(shapeOf);

        const auto topK = std::make_shared<ngraph::opset5::TopK>(dsr, builtSubgraph, 0, "max", "value");

        ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(topK->output(0)),
                                     std::make_shared<ngraph::opset5::Result>(topK->output(1))};

        const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{upperBoundData, realData}, "TopKPropagationOfK");

        const auto transformations = vpu::Transformations{{topK->type_info, vpu::dynamicToStaticShapeTopK}};
        ASSERT_NO_THROW(vpu::DynamicToStaticShape(transformations).run_on_function(function));
        validate(*function);
    }

protected:
    std::shared_ptr<ngraph::Node> buildSubgraph(std::shared_ptr<ngraph::Node> node) const override {
        return std::make_shared<ngraph::opset5::Gather>(
            node,
            ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0}),
            ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0}));
    }
};

TEST_P(DynamicToStaticTopKPropagationShapeOfBased, KPropagation) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticTopKPropagationShapeOfBased, ::testing::ValuesIn(kVec));

}  // namespace
