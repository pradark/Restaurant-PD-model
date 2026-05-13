"""
nlp_txn_extractor.py — Rule-based NLP feature extractor for bank transactions
==============================================================================
Extracts 7 structured features from free-text transaction descriptions
using configurable regex pattern dictionaries. No external NLP dependencies.

Usage
-----
    from skills.nlp_txn_extractor import TransactionNLPExtractor

    extractor = TransactionNLPExtractor()              # default patterns
    df_enriched = extractor.transform(df, col='feature_0')

    # Or extend with custom merchant categories:
    extractor = TransactionNLPExtractor(extra_merchants={'Crypto': [r'\\bCOINBASE\\b']})
    df_enriched = extractor.transform(df, col='description')
"""

import re
import pandas as pd


# ── Default pattern dictionaries ────────────────────────────────────────────

DEFAULT_MERCHANT_PATTERNS: dict[str, list[str]] = {
    "Payroll":      [r"\bPAYROLL\b", r"\bDIRECT\s*DEP(OSIT)?\b",
                     r"\bDD\b.*\b(?:EMPLOYER|SALARY|WAGES)\b", r"\bACH\s*CR\b.*\bPAY\b"],
    "Transfer_ACH": [r"\bACH\s*(?:DEBIT|CREDIT|TRANSFER|PYMT|PMT)\b",
                     r"\bONLINE\s*TRANSFER\b", r"\bBANK\s*TRANSFER\b"],
    "Transfer_P2P": [r"\bVENMO\b", r"\bZELLE\b", r"\bCASH\s*APP\b",
                     r"\bSQUARE\s*CASH\b", r"\bPAYPAL\b"],
    "Subscription": [r"\bNETFLIX\b", r"\bSPOTIFY\b", r"\bHULU\b",
                     r"\bDISNEY\+?\b", r"\bAMAZON\s*PRIME\b",
                     r"\bAPPLE\s*(?:ONE|MUSIC|TV|ICLOUD)\b", r"\bSUBSCRIPTION\b"],
    "Grocery":      [r"\bWHOLE\s*FOODS\b", r"\bKROGER\b", r"\bSAFEWAY\b",
                     r"\bWALMART\b", r"\bTARGET\b", r"\bPUBLIX\b",
                     r"\bGROCERY\b", r"\bSUPERMARKET\b"],
    "Restaurant":   [r"\bMCDONALDS\b", r"\bSTARBUCKS\b", r"\bCHIPOTLE\b",
                     r"\bSUBWAY\b", r"\bPIZZA\b", r"\bRESTAURANT\b", r"\bDINING\b"],
    "FoodDelivery": [r"\bDOORDASH\b", r"\bUBER\s*EATS\b", r"\bGRUBHUB\b",
                     r"\bINSTACART\b", r"\bPOSTMATES\b"],
    "Rideshare":    [r"\bUBER\b(?!\s*EATS)", r"\bLYFT\b"],
    "Retail":       [r"\bAMAZON\b(?!\s*PRIME)", r"\bBEST\s*BUY\b",
                     r"\bHOME\s*DEPOT\b", r"\bETSY\b"],
    "Healthcare":   [r"\bPHARMACY\b", r"\bCVS\b", r"\bWALGREENS\b",
                     r"\bMEDICAL\b", r"\bHOSPITAL\b", r"\bDOCTOR\b"],
    "Entertainment":[r"\bAMC\b", r"\bTICKETMASTER\b", r"\bSTEAM\b",
                     r"\bGAMING\b", r"\bPLAYSTATION\b", r"\bXBOX\b"],
    "Travel":       [r"\bAIRLINE\b", r"\bFLIGHT\b", r"\bHOTEL\b", r"\bAIRBNB\b"],
    "Bank":         [r"\bATM\b", r"\bCASH\s*WITH?DRAW\b", r"\bBANK\s*FEE\b",
                     r"\bOVERDRAFT\b", r"\bCREDIT\s*CARD\s*PAYMENT\b"],
    "Transit":      [r"\bMTA\b", r"\bBART\b", r"\bMETRO\b",
                     r"\bBUS\s*(?:FARE|PASS)\b", r"\bTRANSIT\b"],
    "ATM":          [r"\bATM\s*(?:WITHDRAWAL|WD|CASH)\b"],
}

DEFAULT_CHANNEL_PATTERNS: dict[str, list[str]] = {
    "ACH":    [r"\bACH\b", r"\bAUTOMATED\s*CLEARING\b"],
    "Card":   [r"\bVISA\b", r"\bMASTERCARD\b", r"\bAMEX\b",
               r"\bDEBIT\b", r"\bCREDIT\b", r"\bPURCHASE\b", r"\bPOS\b"],
    "ATM":    [r"\bATM\b"],
    "Wire":   [r"\bWIRE\b", r"\bFED\s*WIRE\b"],
    "Mobile": [r"\bMOBILE\b", r"\bZELLE\b", r"\bVENMO\b", r"\bCASH\s*APP\b"],
    "Check":  [r"\bCHECK\b", r"\bCHEQUE\b"],
}

DEFAULT_RISK_TIERS: dict[str, list[str]] = {
    "High":   ["Transfer_P2P", "ATM", "Entertainment", "Rideshare",
               "Restaurant", "FoodDelivery"],
    "Low":    ["Payroll", "Healthcare", "Grocery", "Transit"],
    "Medium": ["Transfer_ACH", "Subscription", "Retail", "Travel", "Bank"],
}


# ── Extractor class ─────────────────────────────────────────────────────────

class TransactionNLPExtractor:
    """
    Extracts 7 features from free-text transaction descriptions.

    Features extracted
    ------------------
    merchant_category : str   — matched merchant category (or 'Other')
    txn_channel       : str   — payment channel (or 'Other')
    txn_direction     : str   — Debit / Credit / Unknown
    is_recurring      : int   — 1 if RECURRING tag present
    is_p2p            : int   — 1 if P2P transfer detected
    is_international  : int   — 1 if international transaction
    merchant_risk_tier: str   — High / Medium / Low
    """

    def __init__(self,
                 merchant_patterns: dict = None,
                 channel_patterns:  dict = None,
                 risk_tiers:        dict = None,
                 extra_merchants:   dict = None):
        self.merchant_patterns = merchant_patterns or DEFAULT_MERCHANT_PATTERNS
        if extra_merchants:
            self.merchant_patterns = {**self.merchant_patterns, **extra_merchants}
        self.channel_patterns = channel_patterns or DEFAULT_CHANNEL_PATTERNS
        self.risk_tiers       = risk_tiers       or DEFAULT_RISK_TIERS

    def extract_one(self, text) -> dict:
        """Extract features from a single transaction string."""
        if pd.isna(text):
            return self._null_record()
        t = str(text).upper()

        # Merchant category (first match wins)
        mc = "Other"
        for cat, pats in self.merchant_patterns.items():
            if any(re.search(p, t) for p in pats):
                mc = cat
                break

        # Channel (first match wins)
        ch = "Other"
        for chan, pats in self.channel_patterns.items():
            if any(re.search(p, t) for p in pats):
                ch = chan
                break

        # Direction
        direction = "Unknown"
        if any(re.search(p, t) for p in
               [r"\bCREDIT\b", r"\bDEPOSIT\b", r"\bCR\b", r"\bREFUND\b"]):
            direction = "Credit"
        elif any(re.search(p, t) for p in
                 [r"\bDEBIT\b", r"\bPURCHASE\b", r"\bPAYMENT\b",
                  r"\bWITHDRAWAL\b", r"\bPAID\b"]):
            direction = "Debit"

        # Binary flags
        is_recurring     = 1 if re.search(r"\bRECURRING\b", t) else 0
        is_p2p           = 1 if any(re.search(p, t) for p in
                                    [r"\bVENMO\b", r"\bZELLE\b",
                                     r"\bCASH\s*APP\b", r"\bSQUARE\s*CASH\b"]) else 0
        is_international = 1 if any(re.search(p, t) for p in
                                    [r"\bINTL\b", r"\bINTERNATIONAL\b",
                                     r"\bFOREIGN\b", r"\bFX\b"]) else 0

        # Risk tier
        risk_tier = "Medium"
        for tier, cats in self.risk_tiers.items():
            if mc in cats:
                risk_tier = tier
                break

        return {
            "merchant_category": mc,
            "txn_channel":       ch,
            "txn_direction":     direction,
            "is_recurring":      is_recurring,
            "is_p2p":            is_p2p,
            "is_international":  is_international,
            "merchant_risk_tier":risk_tier,
        }

    def transform(self, df: pd.DataFrame, col: str = "feature_0") -> pd.DataFrame:
        """
        Apply extractor to every row of df[col].
        Returns df with 7 new columns appended.
        """
        features = df[col].apply(self.extract_one)
        return pd.concat([df, pd.DataFrame(list(features), index=df.index)], axis=1)

    # ── helpers ──

    def _null_record(self) -> dict:
        return {
            "merchant_category": "Other",
            "txn_channel":       "Other",
            "txn_direction":     "Unknown",
            "is_recurring":      0,
            "is_p2p":            0,
            "is_international":  0,
            "merchant_risk_tier":"Medium",
        }
