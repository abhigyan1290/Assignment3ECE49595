# plain item names for each COCO label we care about
LABEL_QUERIES = {
    "Bag": "bag",
    "Bottle": "water bottle",
    "Glasses": "glasses",
    "Phone": "phone",
    "Watch": "watch",
}

# what students actually type into a lost item report. written by hand, not tuned on the results
REPORT_QUERIES = {
    "Bag": [
        "I lost my backpack in the library",
        "left my bag in the classroom",
        "missing a shoulder bag, probably on a bench",
    ],
    "Bottle": [
        "I left my water bottle in lecture",
        "lost a reusable metal bottle",
        "my drink bottle is missing",
    ],
    "Glasses": [
        "lost my prescription glasses",
        "I can't find my sunglasses",
        "missing eyeglasses with thin frames",
    ],
    "Phone": [
        "I lost my phone in the dining hall",
        "my smartphone fell out of my pocket",
        "missing cell phone with a case",
    ],
    "Watch": [
        "I lost my wristwatch at the gym",
        "left my watch by the sink",
        "missing a smartwatch",
    ],
}
