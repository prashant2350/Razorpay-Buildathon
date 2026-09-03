"""
Real Razorpay signature verification - HMAC-SHA256, implemented from
Razorpay's own docs (not the SDK's built-in helper, so the mechanism is
actually visible rather than hidden behind a library call):
  https://razorpay.com/docs/webhooks/validate-test/

IMPORTANT - Razorpay has TWO different signatures that are easy to confuse
because both are HMAC-SHA256 hex digests, but they use different keys and
sign different data. Verifying one with the other's secret always fails
silently in a way that looks like a bug rather than a security check:

  1. WEBHOOK signature (this is the one that matters for a fraud/risk
     pipeline receiving payment.captured / payment.failed events):
       - arrives in the `X-Razorpay-Signature` request header
       - keyed with your WEBHOOK SECRET (set in Dashboard > Webhooks,
         NOT your API key secret - these are separate values)
       - computed over the RAW webhook request body (not re-serialized JSON -
         re-serializing can reorder keys or change whitespace and silently
         break verification)

  2. PAYMENT / checkout signature (relevant if you also handle Checkout's
     client-side success callback, not webhooks):
       - arrives as `razorpay_signature` from Checkout's success handler
       - keyed with your API KEY SECRET (not the webhook secret)
       - computed over `razorpay_order_id + "|" + razorpay_payment_id`

Why this matters for a risk pipeline specifically: every downstream
decision in this project (detector, rules engine, audit trail) is only as
trustworthy as the data feeding it. If webhook events aren't signature-
verified, an attacker who discovers the endpoint URL could inject fake
payment.captured events directly, bypassing Razorpay entirely and feeding
the detector fabricated "normal" traffic to mask real fraud, or fabricated
fraud signals to trigger false SOFT_HOLDs against a competitor's account.
Signature verification is what makes "the data this pipeline sees actually
came from Razorpay" a checkable fact rather than an assumption.
"""

import hmac
import hashlib
import os


class SignatureVerificationError(Exception):
    pass


def verify_webhook_signature(raw_body: str, received_signature: str, webhook_secret: str = None) -> bool:
    """Verifies an incoming webhook's X-Razorpay-Signature header.

    `raw_body` MUST be the exact raw request body string Razorpay sent -
    not a dict, not json.dumps() of a parsed-and-re-serialized body. In a
    real FastAPI handler this means reading `await request.body()` BEFORE
    calling `await request.json()`, since the raw bytes are what Razorpay
    actually signed.

    Uses hmac.compare_digest (constant-time comparison) rather than `==`,
    which matters here: a naive string comparison leaks timing information
    about how many leading characters matched, which is a known side-channel
    for forging a valid signature byte-by-byte over many requests.
    """
    secret = webhook_secret or os.environ.get("RAZORPAY_WEBHOOK_SECRET")
    if not secret:
        raise SignatureVerificationError(
            "No webhook secret configured. Set RAZORPAY_WEBHOOK_SECRET (from "
            "Dashboard > Settings > Webhooks - this is a value YOU set when "
            "creating the webhook, distinct from your API Key Secret)."
        )

    expected_signature = hmac.new(
        key=secret.encode("utf-8"),
        msg=raw_body.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected_signature, received_signature)


def verify_payment_signature(order_id: str, payment_id: str, received_signature: str, key_secret: str = None) -> bool:
    """Verifies Razorpay Checkout's client-side success callback signature
    (`razorpay_signature`). This is a DIFFERENT mechanism from the webhook
    signature above - different key (API key secret, not webhook secret),
    different signed payload (order_id|payment_id, not the request body).

    This matters for this project specifically because
    dashboard/manual_checkout.html (used in Day 2's Razorpay integration)
    receives exactly this callback shape from Checkout - this function is
    what a production version of that flow would call before trusting the
    payment_id it received."""
    secret = key_secret or os.environ.get("RAZORPAY_KEY_SECRET")
    if not secret:
        raise SignatureVerificationError(
            "No API key secret configured. Set RAZORPAY_KEY_SECRET (your "
            "API Key Secret - distinct from the webhook secret)."
        )

    payload = f"{order_id}|{payment_id}"
    expected_signature = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected_signature, received_signature)
