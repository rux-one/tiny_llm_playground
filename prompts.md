I need to generate a set of information about a fictional company. It's going to be used to train a simple model. The dataset needs to be a single text field where each line follows the scheme: `User: <question about company>\nAssistant: <short response with info>`

Examples:
- User: What area is this company working in?\nAssistant: Our company specializes in making bad software
- User: How many employeees does this company have?\nAssistant: About two hundred permanent employees currently

---

You are tasked with generating a dataset of questions and answers about a fictional company. This dataset will be used to train a simple model. Your goal is to create realistic and varied question-answer pairs that provide information about the company.

Here are the key details about the fictional company:

Company Name: <company_name>{{COMPANY_NAME}}</company_name>
Industry: <industry>{{INDUSTRY}}</industry>
Number of Employees: <employee_count>{{EMPLOYEE_COUNT}}</employee_count>
Founding Year: <founding_year>{{FOUNDING_YEAR}}</founding_year>

Generate questions and answers following this format:
User: [Question about the company]
Assistant: [Short response with information]

Guidelines for generating questions and answers:
1. Questions should cover various aspects of the company, such as its products/services, history, size, culture, leadership, and industry position.
2. Answers should be brief and to the point, typically one or two sentences.
3. Use a conversational tone for both questions and answers.
4. Ensure consistency across all answers regarding company details.
5. Include some questions that can be directly answered using the provided information, and others that require reasonable extrapolation.
6. Vary the complexity and specificity of questions.

Output your dataset as a single block of text, with each question-answer pair on consecutive lines. Do not include any additional formatting or numbering.

Generate 20 question-answer pairs for this dataset.

Begin generating the dataset now.

---

You are tasked with generating a dataset of basic math calculations for training a small language model. This dataset should consist of simple arithmetic problems and their solutions.

Generate {{NUM_EXAMPLES}} examples of math calculations. Each example should include a question (prefixed with "User:") and its corresponding answer (prefixed with "Assistant:").

Follow these guidelines when generating the math problems:
1. Use basic arithmetic operations: addition, subtraction, multiplication, and division.
2. Include a mix of single-operation and multi-operation problems.
3. Use whole numbers ranging from 0 to 1000.
4. Ensure that division problems result in whole numbers (no remainders or decimals).
5. Vary the complexity of the problems, but keep them suitable for basic arithmetic skills.

The output should be formatted as follows:
```
User: [Math Problem]
Assistant: [Correct Answer]
```

Generate the dataset by repeating this format {{NUM_EXAMPLES}} times, each with a unique math problem and its correct solution. Ensure that there is an empty line between each example.

Remember to double-check your calculations to ensure all answers are correct, as accuracy is crucial for training the language model.

Begin generating the dataset now, starting with the first example and continuing until you have created {{NUM_EXAMPLES}} unique problems and solutions.
