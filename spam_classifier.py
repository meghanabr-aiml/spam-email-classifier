import random

print("Spam Email Classifier")

emails = [
    "Win money now",
    "Meeting at 5 PM",
    "Claim your free prize",
    "Project submission tomorrow"
]

for email in emails:
    if "win" in email.lower() or "prize" in email.lower():
        print(email, "-> Spam")
    else:
        print(email, "-> Not Spam")
