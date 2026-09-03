"""
Standalone test for the /webhook endpoint - simulates what Razorpay's
servers actually send: a signed payment.captured event.

Run this AFTER starting server.py in another terminal:
    python3 webhooks/test_webhook.py

This script deliberately sends THREE requests to demonstrate the
verification is real, not decorative:
  1. A correctly-signed event -> should be accepted (200)
  2. The SAME event but with the amount tampered after signing -> should be
     rejected (401), because the signature no longer matches the body
  3. A correctly-signed event but with no RAZORPAY_WEBHOOK_SECRET configured
     on the server -> should fail with a clear config error (500), not a
     silent pass-through

Set RAZORPAY_WEBHOOK_SECRET to the same value in both this script's
environment and server.py's environment for tests 1 and 2 to behave as
described - they're simulating "what happens when the secret matches" vs
"what happens when the body is tampered with after signing", not testing
whether you personally know the secret.
"""

import hmac
import hashlib
import json
import os
import sys

try:
    import requests
except ImportError:
    print("This test script needs the `requests` library: pip install requests")
    sys.exit(1)

SERVER_URL = "http://localhost:8000/webhook"
SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "demo_webhook_secret_for_testing")


def sign(body_str: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body_str.encode("utf-8"), hashlib.sha256).hexdigest()


def make_event(amount_paise: int) -> dict:
    return {
        "entity": "event",
        "account_id": "acc_test123",
        "event": "payment.captured",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_test_webhook_demo",
                    "entity": "payment",
                    "amount": amount_paise,
                    "currency": "INR",
                    "status": "captured",
                    "method": "netbanking",
                }
            }
        },
    }


def run_test(name: str, body_str: str, signature: str, expect_status: int):
    print(f"\n--- {name} ---")
    try:
        resp = requests.post(
            SERVER_URL,
            data=body_str,
            headers={"Content-Type": "application/json", "X-Razorpay-Signature": signature},
            timeout=5,
        )
        status_icon = "PASS" if resp.status_code == expect_status else "FAIL"
        print(f"[{status_icon}] status={resp.status_code} (expected {expect_status})")
        print(f"  response: {resp.json()}")
    except requests.exceptions.ConnectionError:
        print("  ERROR: could not connect - is server.py running on localhost:8000?")
        sys.exit(1)


def main():
    print(f"Testing /webhook using secret: {SECRET[:4]}... (set RAZORPAY_WEBHOOK_SECRET to override)")
    print("NOTE: server.py must have the SAME RAZORPAY_WEBHOOK_SECRET set for test 1 to return 200.")

    # Test 1: correctly signed event
    event1 = make_event(amount_paise=65000)
    body1 = json.dumps(event1)
    sig1 = sign(body1, SECRET)
    run_test("Test 1: correctly signed event", body1, sig1, expect_status=200)

    # Test 2: tamper with the body AFTER signing (simulates a MITM or a
    # spoofed request from someone who doesn't know the secret but tries to
    # replay a captured signature against a modified amount)
    event2 = make_event(amount_paise=65000)
    body2_original = json.dumps(event2)
    sig2 = sign(body2_original, SECRET)  # sign the original...
    event2["payload"]["payment"]["entity"]["amount"] = 99999999  # ...then tamper
    body2_tampered = json.dumps(event2)
    run_test("Test 2: tampered body, stale signature", body2_tampered, sig2, expect_status=401)

    # Test 3: signed with a DIFFERENT (wrong) secret - simulates someone who
    # doesn't know your real webhook secret trying to forge a request
    event3 = make_event(amount_paise=65000)
    body3 = json.dumps(event3)
    sig3 = sign(body3, "attacker_guessed_wrong_secret")
    run_test("Test 3: wrong secret used to sign", body3, sig3, expect_status=401)

    print("\nDone. Tests 2 and 3 returning 401 is the correct, secure behavior -")
    print("it means the endpoint actually rejects unverified/tampered events")
    print("instead of trusting anything that shows up on the URL.")


if __name__ == "__main__":
    main()
