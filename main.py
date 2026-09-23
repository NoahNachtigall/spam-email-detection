import email
from email.header import decode_header
import imaplib
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load credentials
import config

MODEL_FILE = "spam_model.joblib"
VECTORIZER_FILE = "vectorizer.joblib"


def get_or_train_model():
    # Load model from disk if available
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        print("Loading AI model from disk...")
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        print("Model loaded successfully!")
        return model, vectorizer

    print("No saved model found. Training new AI model...")

    # Load dataset automatically detecting column separators
    data = pd.read_csv("spam.csv", sep=None, engine="python", encoding="latin-1")
    data = data.iloc[:, :2]
    data.columns = ["label", "text"]

    # Convert labels: ham -> 0, spam -> 1
    data["label"] = data["label"].map({"ham": 0, "spam": 1})

    # Split dataset
    X_train, _, y_train, _ = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=42
    )

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Save trained artifacts
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("Model trained and saved to disk!")

    return model, vectorizer


def classify_text(model, vectorizer, text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]

    confidence = probabilities[prediction] * 100
    label = "SPAM 🚨" if prediction == 1 else "NOT SPAM 🟢"
    return label, confidence, prediction


def check_unread_emails(model, vectorizer):
    print("\n--- Connecting to Mail Server... ---")
    mail = imaplib.IMAP4_SSL(config.IMAP_SERVER)
    mail.login(config.EMAIL_ACCOUNT, config.APP_PASSWORD)
    mail.select("inbox")

    # Search unread mails
    _, messages = mail.search(None, "UNSEEN")
    email_ids = messages[0].split()

    total_scanned = len(email_ids)
    print(f"Found {total_scanned} unseen emails\n")

    spam_ids = []
    spam_count = 0
    ham_count = 0

    for e_id in email_ids:
        _, msg_data = mail.fetch(e_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                # Decode subject
                subject_header = msg["Subject"]
                if subject_header:
                    subject, encoding = decode_header(subject_header)[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")
                else:
                    subject = "(No Subject)"

                # Extract text body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(
                                "utf-8", errors="ignore"
                            )
                            break
                else:
                    body = msg.get_payload(decode=True).decode(
                        "utf-8", errors="ignore"
                    )

                # Classify email
                full_text = f"{subject} {body}"
                result, confidence, pred_code = classify_text(
                    model, vectorizer, full_text
                )

                if pred_code == 1:
                    spam_count += 1
                    spam_ids.append(e_id)
                else:
                    ham_count += 1

                print(f"SUBJECT: {subject}")
                print(f"RESULT:  {result} ({confidence:.1f}% confidence)\n")

    # Summary report
    print("========================================")
    print("           SUMMARY REPORT               ")
    print("========================================")
    print(f"Total Processed : {total_scanned}")
    print(f"Legitimate (Ham): {ham_count} 🟢")
    print(f"Spam Detected   : {spam_count} 🚨")
    print("========================================\n")

    # Optional spam deletion
    if spam_ids:
        choice = input("Do you want to delete detected spam emails? (y/n): ").strip().lower()
        if choice == "y":
            for s_id in spam_ids:
                mail.store(s_id, "+FLAGS", "\\Deleted")
            mail.expunge()
            print(f"Successfully deleted {len(spam_ids)} spam email(s)!")
        else:
            print("No emails were deleted.")

    mail.logout()


if __name__ == "__main__":
    ai_model, ai_vectorizer = get_or_train_model()
    check_unread_emails(ai_model, ai_vectorizer)
