from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

TEMPLATES = {
    "negative": [
        "My order arrived damaged and support has not replied.",
        "The app keeps crashing whenever I try to checkout.",
        "Terrible experience, refund still pending after one week.",
        "Agent was rude and the ticket was closed without resolution.",
        "Delivery is late again and tracking is completely wrong.",
        "I am extremely unhappy with the product quality.",
        "Charged twice for the same invoice, please fix this.",
        "Package was empty when it arrived, this is unacceptable.",
        "I have been on hold for forty minutes with no answer.",
        "The replacement part does not fit the model I ordered.",
        "Payment failed three times and my card was still charged.",
        "Customer service ignored my previous emails entirely.",
        "Screen cracked within a day of unboxing, very poor QC.",
        "Subscription auto-renewed without any notification.",
        "Driver left the parcel at the wrong address again.",
        "Warranty claim was rejected with no clear explanation.",
        "App login loop keeps locking me out of my account.",
        "Received a used item sold as new, requesting full refund.",
        "Shipping estimate was two days and it took two weeks.",
        "Support promised a callback yesterday and never called.",
        "Product smells of chemicals and I cannot use it indoors.",
        "Invoice shows wrong tax amount and support will not correct it.",
        "Chat bot keeps repeating the same useless scripted reply.",
        "Item was missing accessories listed in the product page.",
        "Night shift delivery woke the whole building, very inconsiderate.",
        "I feel scammed by the flash sale pricing that never applied.",
        "Battery drains in under two hours despite full charge.",
        "Return label was never emailed after I opened the RMA.",
        "Technician cancelled the visit without telling me.",
        "Color and size do not match what I selected at checkout.",
        "Loyalty points vanished after the recent account migration.",
        "Store manager refused to honor the published discount.",
        "Noise from the device is unbearable even at low volume.",
        "Data export feature crashed and corrupted my file.",
        "I was billed for a plan I cancelled last month.",
    ],
    "neutral": [
        "Can you confirm the warranty period for this product?",
        "I need help updating my billing address.",
        "Please share the status of ticket related to invoice copy.",
        "Where can I find documentation for API rate limits?",
        "I want to change the delivery slot for tomorrow.",
        "Looking for details about return policy window.",
        "Need clarification on subscription renewal date.",
        "What are the supported payment methods in my region?",
        "Could you send the tracking number for order TKT-4412?",
        "Is there a student discount available for annual plans?",
        "Please confirm whether this model works with Android 14.",
        "How do I export my chat history from the mobile app?",
        "I need the GST invoice for accounting this quarter.",
        "What is the lead time for spare part SKU-8821?",
        "Can the delivery be redirected to my office address?",
        "Does the premium plan include priority support chat?",
        "Please list the steps to reset my two-factor authentication.",
        "I am checking if weekend delivery is available in 560001.",
        "Where do I upload screenshots for the open support case?",
        "Is the blue variant currently in stock for size medium?",
        "Kindly clarify the difference between basic and plus tiers.",
        "I need the serial number location on the packaging.",
        "Can you confirm the store hours for the MG Road outlet?",
        "Please advise how long account verification usually takes.",
        "What documents are required for a corporate purchase order?",
        "I want to know if gift wrapping is offered at checkout.",
        "How can I pause my subscription for the next billing cycle?",
        "Is international shipping enabled for postal code 10115?",
        "Please share the FAQ link for password recovery.",
        "Does the device support dual SIM and eSIM together?",
        "I am requesting a copy of the signed service agreement.",
        "What is the expected response time for priority tickets?",
        "Can multiple users share one business account seat?",
        "Please confirm the cutoff time for same-day dispatch.",
        "I need guidance on migrating data from the old portal.",
    ],
    "positive": [
        "Support resolved my issue quickly, thank you!",
        "Great product quality and packaging was perfect.",
        "The new chat assistant answered my question instantly.",
        "Happy with the replacement, excellent service.",
        "Fast delivery and the item matches the description.",
        "Very satisfied with the quick refund process.",
        "Awesome experience, will recommend to friends.",
        "Agent went above and beyond to fix my billing error.",
        "Installation guide was clear and setup took minutes.",
        "Love the updated UI, navigation feels much smoother.",
        "Courier was polite and delivered earlier than promised.",
        "Warranty replacement arrived in two days, impressive.",
        "Customer care followed up until everything was working.",
        "Battery life improved a lot after the latest firmware.",
        "Appreciate the proactive email about the shipping delay.",
        "Product feels premium and performs better than expected.",
        "Live chat closed my ticket in under ten minutes.",
        "Thank you for the courtesy upgrade to express shipping.",
        "Return process was painless and refund hit my account fast.",
        "Store staff were knowledgeable and helped me choose well.",
        "I am impressed with how transparent the ETA updates are.",
        "Great value for money compared with the previous model.",
        "The onboarding tutorial made the features easy to learn.",
        "Support engineer explained the root cause clearly.",
        "Packaging was eco-friendly and the item was spotless.",
        "Delighted with the loyalty rewards applied at checkout.",
        "Night support team fixed the outage before morning.",
        "Quality control seems solid, no defects on arrival.",
        "I enjoyed the personalized recommendations in the app.",
        "Refund communication was timely and easy to understand.",
        "Hardware build feels durable after a month of daily use.",
        "Kudos to the team for resolving the double-charge issue.",
        "Seamless account migration with zero data loss.",
        "Positive experience overall from order to delivery.",
        "Will definitely purchase again based on this service.",
    ],
}

PREFIXES = [
    "",
    "Hi team, ",
    "Hello, ",
    "Regarding my account: ",
    "Follow-up: ",
    "Ticket update — ",
    "For order context: ",
]

SUFFIXES = [
    "",
    " please help",
    " asap",
    " thanks",
    "!!",
    " FYI",
    " Looking forward to your reply.",
    " Case ID already shared earlier.",
    " This is my second message.",
    " Kindly escalate if needed.",
]


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


AMBIGUOUS = [
    "The product arrived on time but the packaging looked worn.",
    "Support replied quickly yet the fix did not fully work.",
    "Delivery was fine overall though tracking updates lagged.",
    "I like the design but I am unsure about long-term durability.",
    "Refund started, still waiting to see when money posts.",
    "App works for me most days with occasional slow screens.",
    "Agent was polite; resolution steps were hard to follow.",
    "Quality is okay for the price, not amazing not terrible.",
]


def _maybe_typo(rng: np.random.Generator, text: str) -> str:
    if rng.random() > 0.2 or len(text) < 12:
        return text
    chars = list(text)
    i = int(rng.integers(1, len(chars) - 1))
    op = int(rng.integers(0, 3))
    if op == 0 and chars[i].isalpha():
        chars[i] = chars[i].swapcase()
    elif op == 1:
        chars.insert(i, chars[i])
    else:
        del chars[i]
    return "".join(chars)


def generate_tickets(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = list(TEMPLATES.keys())
    rows = []
    for i in range(n):
        label = labels[int(rng.integers(0, len(labels)))]
        if rng.random() < 0.12:
            text = AMBIGUOUS[int(rng.integers(0, len(AMBIGUOUS)))]
        else:
            text = TEMPLATES[label][int(rng.integers(0, len(TEMPLATES[label])))]
            if rng.random() < 0.3:
                other = labels[int(rng.integers(0, len(labels)))]
                mix = TEMPLATES[other][int(rng.integers(0, len(TEMPLATES[other])))]
                text = f"{text} Also noting: {mix}"
            elif rng.random() < 0.25:
                extra = TEMPLATES[label][int(rng.integers(0, len(TEMPLATES[label])))]
                text = f"{text} {extra}"
        prefix = PREFIXES[int(rng.integers(0, len(PREFIXES)))]
        suffix = SUFFIXES[int(rng.integers(0, len(SUFFIXES)))]
        composed = _maybe_typo(rng, f"{prefix}{text}{suffix}".strip())
        final_label = label
        if rng.random() < 0.06:
            final_label = labels[int(rng.integers(0, len(labels)))]
        rows.append(
            {
                "ticket_id": f"TKT{i:06d}",
                "text": composed,
                "channel": rng.choice(["email", "chat", "app"]),
                "label": final_label,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    out_path = ROOT / cfg["data"]["raw_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_tickets(int(cfg["data"]["sample_size"]), int(cfg["data"]["random_seed"]))
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} tickets -> {out_path}")
    print("Raw data is immutable after this step. Do not edit in place.")
    print(df["label"].value_counts().to_string())
    print(f"Unique texts: {df['text'].nunique()}")


if __name__ == "__main__":
    main()
