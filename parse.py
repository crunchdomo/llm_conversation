import re

def clean_conversation(conversation_text):
    turns = re.split(r"(Trainee Chef:|Master Chef:)", conversation_text)

    cleaned_conversation = []

    speaker = None
    for turn in turns:
        turn = turn.strip()
        if turn == "Trainee Chef:":
            speaker = "Trainee Chef"
        elif turn == "Master Chef:":
            speaker = "Master Chef"
        elif turn:  
            turn = turn.strip()
            turn = re.sub(r'^Agent \d+:', '', turn).strip()
            turn = re.sub(r'^-+$', '', turn).strip()
            turn = re.sub(r'^Query:', '', turn).strip()
            turn = re.sub(r'\*|_|`|\[|\]|\(|\)', '', turn).strip()
            turn = re.sub(r'^(Step \d+:)','', turn).strip()

            if speaker:
                cleaned_conversation.append(f"{speaker}: {turn}")
                speaker = None  

    return cleaned_conversation

def save_conversation_to_file(cleaned_conversation, filename="cleaned_conversation.txt"):

    with open(filename, "w", encoding="utf-8") as f:
        for turn in cleaned_conversation:
            f.write(turn + "\n")
    print(f"Cleaned conversation saved to {filename}")


def clean_conversation_from_file(input_filename="conversation_3.1.txt", output_filename="cleaned_conversation_test.txt"):
    try:
        with open(input_filename, "r", encoding="utf-8") as f:
            conversation_text = f.read()

        cleaned_conversation = clean_conversation(conversation_text)
        save_conversation_to_file(cleaned_conversation, output_filename)

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
# Assuming you have a file named "conversation.txt" in the same directory
# and you want to save the cleaned version to "cleaned_conversation.txt"
clean_conversation_from_file()
