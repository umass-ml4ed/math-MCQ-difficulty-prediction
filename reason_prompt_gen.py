import json

instruction = "Given a math question, its correct answer, and three students' answers, \
your goal is first to generate the reasoning steps to reach the correct answer. More importantly, \
consider the answers provided by three students and explain the reasoning steps they used to arrive at their answers. \
If the answer appears to be pure guesswork or it is unlikely that any logical steps would lead to their answers, \
output 'This answer is a placeholder.' Please strictly follow the demonstration format. This is very important!!!"

# you need to add your own demonstration here
demonstration_1 = ""

demonstration_2 = ""

demonstration_3 = ""

demonstration_4 = ""


with open("question_data.json", "r") as f:
    data = json.load(f)

prompts = []
diffciulties = []
for question in data:
    question_text = "Question: " + question["Question"]
    correct_answer = question["Option1"]
    student_1_answer = question["Option2"]
    student_2_answer = question["Option3"]
    student_3_answer = question["Option4"]
    prompt = f"{instruction}{demonstration_1}{demonstration_2}{demonstration_3}{demonstration_4}\n\nTarget:\n{question_text}\
\nCorrect Answer: {correct_answer}\nStudent 1's Answer: {student_1_answer}\n\
Student 2's Answer: {student_2_answer}\nStudent 3's Answer: {student_3_answer}"
    prompts.append(prompt)
    diffciulties.append(question["beta"])

print(prompts[0])

result = {"prompts": prompts, "difficulties": diffciulties}

with open("eedi_reasoning_prompts.json", "w") as f:
    json.dump(result, f, indent=4)