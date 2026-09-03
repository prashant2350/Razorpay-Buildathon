"""
Thin wrapper around the real Razorpay Python SDK (test mode).

Credentials are read ONLY from environment variables - never hardcode a
key_id/key_secret in source. Set them via a local .env (see .env.example)
or your shell:

    export RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxx
    export RAZORPAY_KEY_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx

Install the SDK first:  pip install razorpay

Docs used:
  https://razorpay.com/docs/api/authentication/
  https://razorpay.com/docs/api/orders/
  https://razorpay.com/docs/api/payments/
  https://razorpay.com/docs/payments/payments/test-card-details/
"""

import os

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass  # python-dotenv not installed - env vars must be set manually


class RazorpayNotConfigured(RuntimeError):
    pass


def get_client():
    """Returns an authenticated razorpay.Client, or raises a clear error
    telling the caller exactly what env vars are missing - never silently
    falls back to a fake client."""
    try:
        import razorpay
    except ImportError as e:
        raise RazorpayNotConfigured(
            "razorpay SDK not installed. Run: pip install razorpay"
        ) from e

    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise RazorpayNotConfigured(
            "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set in environment. "
            "Generate test-mode keys at dashboard.razorpay.com "
            "-> Account & Settings -> API Keys, then export them."
        )
    if not key_id.startswith("rzp_test_"):
        raise RazorpayNotConfigured(
            f"Key '{key_id[:12]}...' doesn't look like a TEST mode key "
            "(expected it to start with 'rzp_test_'). Refusing to run against "
            "what might be a live key."
        )

    client = razorpay.Client(auth=(key_id, key_secret))
    return client


def fetch_all_payments(count=100, skip=0):
    """Calls the real Fetch All Payments endpoint. Returns Razorpay's raw
    payment.entity list - https://razorpay.com/docs/api/payments/fetch-all/"""
    client = get_client()
    return client.payment.all({"count": count, "skip": skip})


def fetch_all_orders(count=100, skip=0):
    client = get_client()
    return client.order.all({"count": count, "skip": skip})
