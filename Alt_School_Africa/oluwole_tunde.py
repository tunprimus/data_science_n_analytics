#!/usr/bin/env python3

# This is a programme to calculate the average score for each student

# Define a dictionary to store student data
students = {}

# Function to add a student
def add_student():
    name = input("Enter student name: ")
    math_score = float(input("Enter math score: "))
    science_score = float(input("Enter science score: "))
    english_score = float(input("Enter english score: "))

    # Calculate average score
    average_score = (math_score + science_score + english_score) / 3
    average_score = round(average_score, 1)

    # Assign grade based on average score
    if average_score >= 90:
        grade = "A"
    elif average_score >= 70:
        grade = "B"
    elif average_score >= 50:
        grade = "C"
    elif average_score >= 40:
        grade = "D"
    else:
        grade = "F"

    # Store student data in dictionary
    students[name] = {
        "math_score": math_score,
        "science_score": science_score,
        "english_score": english_score,
        "average_score": average_score,
        "grade": grade
    }
    return students

# Function to display all student data
def display_students():
    for name, data in students.items():
        print(f"Name: {name}")
        print(f"Math Score: {data['math_score']}")
        print(f"Science Score: {data['science_score']}")
        print(f"English Score: {data['english_score']}")
        print(f"Average Score: {data['average_score']}")
        print(f"Grade: {data['grade']}")
        print("------------------------\n")

# Function to calculate average score for each student
def calculate_average_scores():
    if len(students) == 0:
        raise ValueError("No students found")
    else:
        for name, data in students.items():
            average_score = (data["math_score"] + data["science_score"] + data["english_score"]) / 3
            students[name]["average_score"] = round(average_score, 1)
            # students[name]["average_score"] = f"{average_score}:.1f"

# Function to assign grades based on average scores
def assign_grades():
    if len(students) == 0:
        raise ValueError("No students found")
    else:
        for name, data in students.items():
            if data["average_score"] >= 90:
                grade = "A"
            elif data["average_score"] >= 70:
                grade = "B"
            elif data["average_score"] >= 50:
                grade = "C"
            elif data["average_score"] >= 40:
                grade = "D"
            else:
                grade = "F"
            students[name]["grade"] = grade

# Main programme loop
while True:
    print("Student Grade Management System")
    print("1. Add Student")
    print("2. Display Students")
    print("3. Calculate Average Scores")
    print("4. Assign Grades")
    print("5. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        add_student()
    elif choice == "2":
        display_students()
    elif choice == "3":
        calculate_average_scores()
    elif choice == "4":
        assign_grades()
    elif choice == "5":
        break
    else:
        print("Invalid choice. Please try again.")
