from model_handler import respond
from typer import type_text

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = respond(user_input)
    type_text(response)
