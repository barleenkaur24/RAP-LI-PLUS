
from risk_detection.regex_detector import detect_pii
from risk_detection.risk_classifier import classify_entities


text = (
    "My email is barleen@example.com. "
    "My phone number is 9876543210."
)

detected_entities = detect_pii(text)
classified_entities = classify_entities(detected_entities)

print("Final classified entities:")

for entity in classified_entities:
    print(entity)