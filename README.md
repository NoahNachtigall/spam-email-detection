# Spam Filter Email

A small Python script that checks unread emails in Gmail, classifies them as spam or not spam, and prints the result with confidence.

## What was wrong

The app was trying to train a model from `spam.csv`, but that file was not a real dataset. It contained a placeholder page:

```
404: Not Found
```

That made the training data empty, which caused this crash:

```
ValueError: With n_samples=0, test_size=0.2 ... the resulting train set will be empty.
```

I fixed this by validating the CSV and falling back to a built-in sample dataset when the file is missing or invalid.

## Setup

1. Open `config.py` and replace the Gmail values with your own mail account details.
2. Enable 2-Step Verification on your Google account.
3. Generate a Gmail App Password:
   - Go to Google Account > Security
   - Under "Signing in to Google", enable 2-Step Verification
   - Then open "App passwords"
   - Generate a new app password for "Mail"
   - Copy the 16-character password and paste it into `config.py` without spaces
4. Activate the project virtual environment:

```bash
cd /home/luca/programming/python/spam-filter-email
source venv/bin/activate
```

5. Install any missing dependencies:

```bash
pip install joblib pandas scikit-learn
```

## Run it

```bash
python main.py
```

If you use the project venv directly, run:

```bash
./venv/bin/python main.py
```

## Notes

- The script looks for unread emails in the Gmail Inbox.
- It uses Gmail IMAP and needs an app password for the account.
- If `spam.csv` is missing or broken, the script now falls back to a built-in sample dataset so it does not crash.

## Important

Do not commit your real Gmail credentials. Keep them local in `config.py` and avoid pushing them to GitHub.
