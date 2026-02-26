"""
Market pool for deliberate lending scenarios.

Borrowers can remain in-market for multiple weeks and lenders can leave offers
open for multiple weeks.
"""

from __future__ import annotations

from .models import Borrower, MarketBorrower, OpenOffer, TermSheet


class MarketPool:
    def __init__(
        self,
        borrower_patience_weeks: int = 1,
        offer_validity_weeks: int = 1,
    ) -> None:
        if borrower_patience_weeks <= 0:
            raise ValueError("borrower_patience_weeks must be > 0")
        if offer_validity_weeks <= 0:
            raise ValueError("offer_validity_weeks must be > 0")

        self.borrower_patience_weeks = borrower_patience_weeks
        self.offer_validity_weeks = offer_validity_weeks
        self.borrowers: dict[str, MarketBorrower] = {}
        self.current_week: int = 0
        self._offer_counter: int = 0
        self._lender_capital: dict[str, dict] = {}

    def set_current_week(self, week: int) -> None:
        self.current_week = week

    def set_lender_capital_snapshot(
        self,
        lender_id: str,
        *,
        total_capital: float,
        deployed_capital: float,
        available_capital: float,
        cost_of_capital: float,
    ) -> None:
        self._lender_capital[lender_id] = {
            "total_capital": float(total_capital),
            "deployed_capital": float(deployed_capital),
            "available_capital": float(available_capital),
            "cost_of_capital": float(cost_of_capital),
        }

    def add_cohort(self, borrowers: list[Borrower], week: int) -> None:
        self.current_week = week
        patience = max(0, self.borrower_patience_weeks - 1)
        for borrower in borrowers:
            existing = self.borrowers.get(borrower.id)
            if existing and existing.status == "shopping":
                continue
            self.borrowers[borrower.id] = MarketBorrower(
                borrower=borrower,
                entered_week=week,
                weeks_remaining=patience,
            )

    def get_market_borrower(self, borrower_id: str) -> MarketBorrower | None:
        return self.borrowers.get(borrower_id)

    def get_all_shopping_borrowers(self) -> list[Borrower]:
        return [
            mb.borrower
            for mb in self.borrowers.values()
            if mb.status == "shopping" and mb.weeks_remaining >= 0
        ]

    def get_pool_for_lender(self, lender_id: str) -> list[Borrower]:
        return [
            mb.borrower
            for mb in self.borrowers.values()
            if mb.status == "shopping"
            and mb.weeks_remaining >= 0
            and lender_id not in mb.rejected_by
        ]

    def open_offers_for_lender(self, lender_id: str) -> list[OpenOffer]:
        offers: list[OpenOffer] = []
        for mb in self.borrowers.values():
            if mb.status != "shopping":
                continue
            for offer in mb.open_offers:
                if offer.lender_id == lender_id and offer.status == "open":
                    offers.append(offer)
        return offers

    def reserved_capital_for_lender(self, lender_id: str) -> float:
        return sum(
            offer.term_sheet.loan_amount
            for offer in self.open_offers_for_lender(lender_id)
        )

    def tick(self, week: int) -> list[str]:
        self.current_week = week
        events: list[str] = []
        for mb in self.borrowers.values():
            if mb.status != "shopping":
                continue
            mb.weeks_remaining -= 1
            for offer in mb.open_offers:
                if offer.status != "open":
                    continue
                offer.weeks_remaining -= 1
                if offer.weeks_remaining < 0:
                    offer.status = "expired"
                    events.append(
                        f"  [Market] Offer expired: {offer.offer_id} for "
                        f"{mb.borrower.dossier.company_name}"
                    )
        return events

    def resolve_expired(self) -> tuple[list[tuple[MarketBorrower, OpenOffer]], list[MarketBorrower]]:
        accepted: list[tuple[MarketBorrower, OpenOffer]] = []
        expired: list[MarketBorrower] = []

        for mb in self.borrowers.values():
            if mb.status != "shopping" or mb.weeks_remaining >= 0:
                continue

            open_offers = [o for o in mb.open_offers if o.status == "open"]
            if open_offers:
                winner = min(
                    open_offers,
                    key=lambda o: (
                        o.term_sheet.interest_rate,
                        -o.term_sheet.loan_amount,
                        o.issued_week,
                        o.offer_id,
                    ),
                )
                winner.status = "accepted"
                for offer in open_offers:
                    if offer.offer_id != winner.offer_id:
                        offer.status = "expired"
                mb.status = "booked"
                accepted.append((mb, winner))
            else:
                mb.status = "expired"
                expired.append(mb)

        return accepted, expired

    def force_resolve_all(self) -> tuple[list[tuple[MarketBorrower, OpenOffer]], list[MarketBorrower]]:
        for mb in self.borrowers.values():
            if mb.status == "shopping":
                mb.weeks_remaining = -1
        return self.resolve_expired()

    def record_pass(self, lender_id: str, borrower_id: str, week: int) -> None:
        mb = self.borrowers.get(borrower_id)
        if not mb or mb.status != "shopping":
            return
        mb.passed_by[lender_id] = week

    def record_reject(self, lender_id: str, borrower_id: str) -> None:
        mb = self.borrowers.get(borrower_id)
        if not mb or mb.status != "shopping":
            return
        mb.rejected_by.add(lender_id)
        mb.passed_by.pop(lender_id, None)
        for offer in mb.open_offers:
            if offer.lender_id == lender_id and offer.status == "open":
                offer.status = "expired"

    def add_offer(
        self,
        lender_id: str,
        borrower_id: str,
        term_sheet: TermSheet,
        reasoning: str,
        issued_week: int,
        validity_weeks: int | None = None,
    ) -> OpenOffer | None:
        mb = self.borrowers.get(borrower_id)
        if not mb or mb.status != "shopping":
            return None

        for offer in mb.open_offers:
            if offer.lender_id == lender_id and offer.status == "open":
                offer.status = "expired"

        self._offer_counter += 1
        offer = OpenOffer(
            offer_id=f"OFR-{self._offer_counter:04d}",
            lender_id=lender_id,
            borrower_id=borrower_id,
            term_sheet=term_sheet,
            reasoning=reasoning,
            issued_week=issued_week,
            weeks_remaining=max(0, (validity_weeks or self.offer_validity_weeks) - 1),
        )
        mb.open_offers.append(offer)
        return offer

    def remove_booked(self, borrower_id: str) -> None:
        mb = self.borrowers.get(borrower_id)
        if not mb:
            return
        mb.status = "booked"
        for offer in mb.open_offers:
            if offer.status == "open":
                offer.status = "expired"

    def build_pipeline_context(self, lender_id: str) -> dict:
        snapshot = self._lender_capital.get(lender_id, {})
        total_capital = float(snapshot.get("total_capital", 0.0))
        deployed_capital = float(snapshot.get("deployed_capital", 0.0))
        available_capital = float(snapshot.get("available_capital", 0.0))
        cost_of_capital = float(snapshot.get("cost_of_capital", 0.0))
        reserved = self.reserved_capital_for_lender(lender_id)
        available_for_new = max(0.0, available_capital - reserved)

        offers = self.open_offers_for_lender(lender_id)
        offers.sort(key=lambda o: (o.weeks_remaining, o.offer_id))
        open_offers = []
        for offer in offers:
            mb = self.borrowers.get(offer.borrower_id)
            if not mb:
                continue
            open_offers.append({
                "offer_id": offer.offer_id,
                "borrower_id": offer.borrower_id,
                "borrower_name": mb.borrower.dossier.company_name,
                "sector": mb.borrower.dossier.sector,
                "loan_amount": offer.term_sheet.loan_amount,
                "interest_rate": offer.term_sheet.interest_rate,
                "term_months": offer.term_sheet.term_months,
                "weeks_remaining": offer.weeks_remaining,
                "issued_week": offer.issued_week,
                "reasoning": offer.reasoning,
            })

        market_rows = []
        for mb in self.borrowers.values():
            if mb.status != "shopping" or mb.weeks_remaining < 0 or lender_id in mb.rejected_by:
                continue
            borrower = mb.borrower
            lender_open_offer = next(
                (
                    o.offer_id for o in mb.open_offers
                    if o.status == "open" and o.lender_id == lender_id
                ),
                None,
            )
            market_rows.append({
                "borrower_id": borrower.id,
                "company_name": borrower.dossier.company_name,
                "sector": borrower.dossier.sector,
                "request_amount": borrower.dossier.loan_request_amount,
                "entered_week": mb.entered_week,
                "weeks_remaining": mb.weeks_remaining,
                "passed_week": mb.passed_by.get(lender_id),
                "other_open_offer_count": sum(
                    1 for o in mb.open_offers if o.status == "open" and o.lender_id != lender_id
                ),
                "has_your_offer": lender_open_offer is not None,
                "your_offer_id": lender_open_offer,
            })

        new_this_week = [row for row in market_rows if row["entered_week"] == self.current_week]
        returning_after_pass = [
            row for row in market_rows
            if row["entered_week"] < self.current_week and row["passed_week"] is not None
        ]
        with_your_offer = [row for row in market_rows if row["has_your_offer"]]

        return {
            "week": self.current_week,
            "capital": {
                "total_capital": total_capital,
                "deployed_capital": deployed_capital,
                "reserved_open_offers": reserved,
                "available_capital": available_capital,
                "available_for_new_commitments": available_for_new,
                "cost_of_capital_annual": cost_of_capital,
            },
            "open_offers": open_offers,
            "market_pool": {
                "total_shopping": len(market_rows),
                "new_this_week": new_this_week,
                "returning_after_pass": returning_after_pass,
                "with_your_offer": with_your_offer,
                "all": market_rows,
            },
            "actions_available": ["APPROVE", "REJECT", "PASS"],
        }
