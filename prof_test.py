import json

with open("professors.json") as f:
    profs = json.load(f)

print(f"Total professors: {len(profs)}")

has_email = sum(1 for p in profs if p["email"])
has_interests = sum(1 for p in profs if p["research_interests"])
has_bio = sum(1 for p in profs if p["bio"])

print(f"Has email:     {has_email}/116")
print(f"Has interests: {has_interests}/116")
print(f"Has bio:       {has_bio}/116")