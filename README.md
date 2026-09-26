# Gmail Spam Filter

A command-line spam classifier that scans unread Gmail messages and reports whether each looks like spam. It can optionally delete messages classified as spam.

## Requirements

- Python 3.10 or newer
- A Gmail account with IMAP enabled
- Google 2-Step Verification and a Gmail App Password
- An operating-system keyring service (for example, GNOME Keyring or KWallet on Linux)

Use an App Password, not your normal Google password. Google only offers App Passwords for accounts that meet its security requirements; work or school accounts may have this feature disabled by an administrator.

## Install

From this directory, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run

```bash
python main.py
```

On the first run, enter your Gmail address and App Password when prompted. The password prompt does not display typed characters. The credentials are encrypted in `credentials.enc`; the encryption key is stored separately in your operating-system keyring. Later runs decrypt the saved credentials automatically.

If the keyring is unavailable or locked, unlock/configure it and run the program again. Do not replace keyring storage with a key saved beside the encrypted file. To switch accounts, remove `credentials.enc` and run the program again. Removing that file does not revoke the Google App Password.

## What it does

- Scans unread messages in the Gmail inbox over IMAP.
- Trains a Naive Bayes text classifier from `spam.csv` on the first run and saves the model locally.
- Uses a small built-in sample dataset if `spam.csv` is missing, invalid, or does not contain both spam and ham examples. That fallback is only for trying the program; a real training dataset will classify better.
- Asks before deleting messages it classified as spam.

The classifier is a basic demonstration, not a reliable security filter. Review messages before deleting them.

## Tests

```bash
python -m pip install pytest
python -m pytest
```

## Credential safety

Keep `credentials.enc` and your operating-system account private. The encrypted file is ignored by Git, and the encryption key is kept in the OS keyring. Anyone with access to both the encrypted file and your unlocked keyring can recover the credentials. Revoke the App Password in your Google Account if it was exposed or you no longer use it.