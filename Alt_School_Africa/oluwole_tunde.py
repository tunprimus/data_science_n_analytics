#!/usr/bin/env python3

# Define a function to calculate the average score
def calculate_average_score(scores):
    """
    Calculate the average score from a list of scores.

    Args:
        scores (list): A list of scores.

    Returns:
        float: The average score.
    """
    return sum(scores) / len(scores)

# Define a function to assign a grade based on the average score
def assign_grade(average_score):
    """
    Assign a grade based on the average score.

    Args:
        average_score (float): The average score.

    Returns:
        str: The assigned grade.
    """
    if average_score >= 90:
        return "A"
    elif average_score >= 70:
        return "B"
    elif average_score >= 50:
        return "C"
    elif average_score >= 40:
        return "D"
    else:
        return "F"

# Define a function to display student data
def display_student_data(student_data):
    """
    Display student data.

    Args:
        student_data (dict): A dictionary containing student data.
    """
    for student, data in student_data.items():
        print(f"Name: {student}")
        print(f"Scores: {data['scores']}")
        print(f"Average Score: {data['average_score']}")
        print(f"Grade: {data['grade']}")
        print()

# Define a function to add a student
def add_student(student_data):
    """
    Add a student to the student data dictionary.

    Args:
        student_data (dict): A dictionary containing student data.

    Returns:
        dict: The updated student data dictionary.
    """
    name = input("Enter student name: ")
    scores = []
    for subject in ["Math", "Science", "English"]:
        score = float(input(f"Enter {subject} score: "))
        scores.append(score)
    average_score = calculate_average_score(scores)
    grade = assign_grade(average_score)
    student_data[name] = {
        "scores": scores,
        "average_score": average_score,
        "grade": grade
    }
    return student_data

# Define a function to display the menu
def display_menu():
    """
    Display the menu.
    """
    print("Student Grade Management System")
    print("-------------------------------")
    print("1. Add Student")
    print("2. Display Student Data")
    print("3. Exit")

# Define the main function
def main():
    """
    The main function.
    """
    student_data = {}
    while True:
        display_menu()
        choice = input("Enter your choice: ")
        if choice == "1":
            student_data = add_student(student_data)
        elif choice == "2":
            display_student_data(student_data)
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

# Call the main function
if __name__ == "__main__":
    main()
