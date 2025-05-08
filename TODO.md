# TODO

## Long term

1. 为ChromaDB构建可视化查询能力，便于后续排查问题及演示

## Daily

1. RAG 优化
(可选) 添加过滤: 在重排序后，根据分数过滤掉一部分结果。
(可选) 添加去重: 在过滤后，对剩余的块进行基于 embedding 的去重。
(可选) 尝试重组: 修改 PromptBuilder._format_context，实现 "Reverse" 或 "Sides" 排列策略。

## To Optimize

// ds 模型的非准确性
1. 2025-04-25 14:22:17,759 - rag.tokenizer_util - WARNING - Model 'deepseek-chat' not found by tiktoken. Falling back to default encoding 'cl100k_base'. Token counts may be inaccurate.
