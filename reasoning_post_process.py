import json
import re

with open("eedi_generated_reasonings_4o.json", "r") as f:
    data = json.load(f)

prompts = data["prompts"]
predictions = data["predictions"]
gt_difficulties = data["gt_difficulties"]

question_stems = []
correct_answers = []
student_1_answers = []
student_2_answers = []
student_3_answers = []
correct_answer_reasonings = []
student_1_reasonings = []
student_2_reasonings = []
student_3_reasonings = []

for prompt, reasoning in zip(prompts, predictions):
    prompt = prompt.split("Target:\n")[1]
    print("prompt:", prompt)
    print("reasoning:", reasoning)
    question_stem = prompt.split("Question:")[1].split("\nCorrect Answer:")[0].strip()   
    question_stems.append(question_stem)
    correct_answer = prompt.split("\nCorrect Answer: ")[1].split("\nStudent 1's Answer:")[0].strip()
    correct_answers.append(correct_answer)
    student_1_answer = prompt.split("\nStudent 1's Answer: ")[1].split("\nStudent 2's Answer:")[0].strip()
    student_1_answers.append(student_1_answer)
    student_2_answer = prompt.split("\nStudent 2's Answer: ")[1].split("\nStudent 3's Answer:")[0].strip()
    student_2_answers.append(student_2_answer)
    student_3_answer = prompt.split("\nStudent 3's Answer: ")[1].strip()
    student_3_answers.append(student_3_answer)
    correct_answer_reasoning = reasoning.split("Reasoning steps for the correct answer:")[1].split("Reasoning steps for student 1's answer:")[0].strip()
    correct_answer_reasonings.append(correct_answer_reasoning)
    student_1_reasoning = reasoning.split("Reasoning steps for student 1's answer:")[1].split("Reasoning steps for student 2's answer:")[0].strip()
    student_1_reasonings.append(student_1_reasoning)
    student_2_reasoning = reasoning.split("Reasoning steps for student 2's answer:")[1].split("Reasoning steps for student 3's answer:")[0].strip()
    student_2_reasonings.append(student_2_reasoning)
    student_3_reasoning = reasoning.split("Reasoning steps for student 3's answer:")[1].strip()
    student_3_reasonings.append(student_3_reasoning)

# store each question data in a dictionary
question_data = []
for i in range(len(question_stems)):
    question_data.append({
        "QuestionStem": question_stems[i],
        "CorrectAnswer": correct_answers[i],
        "Student1Answer": student_1_answers[i],
        "Student2Answer": student_2_answers[i],
        "Student3Answer": student_3_answers[i],
        "CorrectAnswerReasoning": correct_answer_reasonings[i],
        "Student1Reasoning": student_1_reasonings[i],
        "Student2Reasoning": student_2_reasonings[i],
        "Student3Reasoning": student_3_reasonings[i],
        "GroundTruthDifficulty": gt_difficulties[i]
    })

# write question data to a json file
with open("eedi_math_reasoning_data.json", "w") as f:
    json.dump(question_data, f, indent=4)

