from scripts.build_process_workflows import _validate_card


def _card(step: str) -> dict:
    return {
        "title": "book a flight",
        "applies_when": "the user approves a flight",
        "preconditions": [],
        "steps": [step],
        "branches": [],
        "avoid": [],
        "keywords": ["book flight"],
    }


def test_validate_card_accepts_tool_prefixed_keyword_arguments() -> None:
    assert _validate_card(
        _card(
            "create_booking(flight_id=<id>, add_wifi=<bool>, "
            "add_extra_legroom=<bool>, add_insurance=<bool>)"
        ),
        {"create_booking"},
    )


def test_validate_card_rejects_hallucinated_tool_call_or_instruction() -> None:
    assert not _validate_card(
        _card("create_booking(flight_id=<id>), then add_baggage(booking_id=<id>)"),
        {"create_booking"},
    )
    assert not _validate_card(
        _card("Use add_baggage after create_booking(flight_id=<id>)."),
        {"create_booking"},
    )
