import getpass
import email
import json
import imaplib
import os
from pathlib import Path

import keyring
import joblib
import pandas as pd
from cryptography.fernet import Fernet, InvalidToken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "spam_model.joblib"
VECTORIZER_FILE = BASE_DIR / "vectorizer.joblib"
DATASET_FILE = BASE_DIR / "spam.csv"
CREDENTIALS_FILE = BASE_DIR / "credentials.enc"
IMAP_SERVER = "imap.gmail.com"
KEYRING_SERVICE = "spam-email-detection"
KEYRING_USERNAME = "credentials-encryption-key"

FALLBACK_TRAINING_DATA = [
    ("ham", "Are we still meeting for lunch today?"),
    ("ham", "Please review the project notes before our call."),
    ("ham", "Your appointment is confirmed for tomorrow morning."),
    ("ham", "I sent the photos from the weekend. Let me know what you think."),
    ("spam", "You won a free prize. Claim your cash reward now."),
    ("spam", "Urgent! Claim your exclusive offer by clicking this link."),
    ("spam", "Congratulations, you have been selected for a free gift."),
    ("spam", "Earn money fast with this limited time opportunity."),
]


def load_training_data(path="spam.csv"):
    """Load the SMS spam dataset, falling back when the file is unusable."""
    try:
        data = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
        if data.shape[1] < 2:
            raise ValueError("The dataset must contain a label and a text column.")

        data = data.iloc[:, :2].copy()
        data.columns = ["label", "text"]
        data["label"] = data["label"].astype(str).str.strip().str.lower()
        data["label"] = data["label"].map({"ham": 0, "spam": 1})
        data = data.dropna(subset=["label", "text"])
        data["text"] = data["text"].astype(str)
        if data.empty or data["label"].nunique() < 2:
            raise ValueError("The dataset must contain both ham and spam examples.")
        return data
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
        return pd.DataFrame(FALLBACK_TRAINING_DATA, columns=["label", "text"]).assign(
            label=lambda rows: rows["label"].map({"ham": 0, "spam": 1})
        )


def _get_encryption_key(create=False):
    try:
        key = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        if key is None and create:
            key = Fernet.generate_key().decode("ascii")
            keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, key)
    except Exception as error:
        raise RuntimeError(
            "Could not access the operating-system keyring. Unlock or configure your "
            "keyring service, then try again."
        ) from error
    if key is None:
        raise RuntimeError(
            "The encryption key is missing from your OS keyring. Remove credentials.enc "
            "and run the program again to set credentials up."
        )
    return key.encode("ascii")


def save_credentials(credentials, path=CREDENTIALS_FILE):
    key = _get_encryption_key(create=True)
    encrypted = Fernet(key).encrypt(json.dumps(credentials).encode("utf-8"))
    path = Path(path)
    path.write_bytes(encrypted)
    path.chmod(0o600)


def decrypt_credentials(path=CREDENTIALS_FILE):
    """Decrypt saved credentials using the key stored in the OS keyring."""
    path = Path(path)
    try:
        encrypted = path.read_bytes()
        plaintext = Fernet(_get_encryption_key()).decrypt(encrypted)
        credentials = json.loads(plaintext.decode("utf-8"))
    except InvalidToken as error:
        raise RuntimeError(
            "Saved credentials could not be decrypted. Remove credentials.enc and "
            "run the program again to enter them."
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Saved credentials could not be read.") from error

    if not credentials.get("email") or not credentials.get("app_password"):
        raise RuntimeError("Saved credentials are incomplete. Remove credentials.enc and retry.")
    return credentials


def get_credentials():
    if CREDENTIALS_FILE.exists():
        return decrypt_credentials()

    print("Enter your Gmail details. The app password will not be shown as you type.")
    email_address = input("Gmail address: ").strip()
    app_password = getpass.getpass("Gmail app password: ").replace(" ", "")
    if "@" not in email_address or not app_password:
        raise ValueError("Enter a valid email address and a non-empty app password.")

    credentials = {"email": email_address, "app_password": app_password}
    save_credentials(credentials)
    print("Credentials saved encrypted on this device.")
    return credentials


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
    data = load_training_data(DATASET_FILE)

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
    credentials = get_credentials()
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    try:
        mail.login(credentials["email"], credentials["app_password"])
    except imaplib.IMAP4.error as error:
        raise RuntimeError(
            "Gmail login failed. Check the address, enable IMAP in Gmail, and use an "
            "App Password (not your regular account password)."
        ) from error

    try:
        status, _ = mail.select("inbox")
        if status != "OK":
            raise RuntimeError("Could not open the Gmail inbox.")

        status, messages = mail.search(None, "UNSEEN")
        if status != "OK":
            raise RuntimeError("Could not search for unread Gmail messages.")
        email_ids = messages[0].split() if messages else []

        total_scanned = len(email_ids)
        print(f"Found {total_scanned} unseen emails\n")

        spam_ids = []
        spam_count = 0
        ham_count = 0

        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            if status != "OK":
                continue
            for response_part in msg_data:
                if not isinstance(response_part, tuple):
                    continue
                msg = email.message_from_bytes(response_part[1])

                subject = str(msg.get("Subject", "(No Subject)"))

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain" and not part.get_filename():
                            payload = part.get_payload(decode=True) or b""
                            body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                            break
                else:
                    payload = msg.get_payload(decode=True) or b""
                    body = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")

                full_text = f"{subject} {body}"
                result, confidence, pred_code = classify_text(
                    model, vectorizer, full_text
                )

                if pred_code == 1:
                    spam_count += 1
                    spam_ids.append(email_id)
                else:
                    ham_count += 1

                print(f"SUBJECT: {subject}")
                print(f"RESULT:  {result} ({confidence:.1f}% confidence)\n")

        print("========================================")
        print("           SUMMARY REPORT               ")
        print("========================================")
        print(f"Total Processed : {total_scanned}")
        print(f"Legitimate (Ham): {ham_count}")
        print(f"Spam Detected   : {spam_count}")
        print("========================================\n")

        if spam_ids:
            choice = input("Delete detected spam emails? (y/n): ").strip().lower()
            if choice == "y":
                for spam_id in spam_ids:
                    mail.store(spam_id, "+FLAGS", "\\Deleted")
                mail.expunge()
                print(f"Successfully deleted {len(spam_ids)} spam email(s)!")
            else:
                print("No emails were deleted.")
    finally:
        try:
            mail.logout()
        except imaplib.IMAP4.error:
            pass


if __name__ == "__main__":
    ai_model, ai_vectorizer = get_or_train_model()
    check_unread_emails(ai_model, ai_vectorizer)
