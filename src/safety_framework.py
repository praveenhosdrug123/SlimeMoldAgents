# src/safetyframework.py
from dataclasses import dataclass
from typing import List, Set, Tuple
import re
from datetime import datetime
from utils.debug_utils import DebugUtils

@dataclass
class SafetyViolation:
    type: str
    description: str
    detected_pattern: str

class SafetyFramework:
    def __init__(self):
        self.logger = DebugUtils.get_logger(self.__class__.__name__)
        self.debug_utils = DebugUtils()
        
        # Initialize pattern dictionaries
        self.initialize_safety_patterns()
        self.logger.info("SafetyFramework initialized successfully")

    @DebugUtils.trace_function
    def initialize_safety_patterns(self):
        """Initialize all safety check patterns"""
        self.logger.debug("Initializing safety patterns")
        
        self.adversarial_patterns = {
            'prompt_injection': r"(?i)(prompt|system|instruction).*:.*",
            'code_injection': r"(?i)(exec|eval|system|import os|import sys)",
            'sql_injection': r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE)",
            'xss_attempt': r"(?i)(<script|javascript:|onclick=|onerror=)",
            'delimiter_manipulation': r"(?i)(<|>|{|}|\[|\])"
        }
        self.logger.debug(f"Loaded {len(self.adversarial_patterns)} adversarial patterns")

        self.pii_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b\d{16}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'address': r'\b\d+\s+([A-Za-z]+ )+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b'
        }
        self.logger.debug(f"Loaded {len(self.pii_patterns)} PII patterns")

        self.harmful_content_patterns = {
            'hate_speech': r"(?i)(hate|racial slur|discriminat)",
            'violence': r"(?i)(kill|murder|attack|bomb|weapon)",
            'self_harm': r"(?i)(suicide|self-harm|cut myself)",
            'harassment': r"(?i)(harass|bully|threaten|stalk)",
            'explicit_content': r"(?i)(explicit|nude|porn)"
        }
        self.logger.debug(f"Loaded {len(self.harmful_content_patterns)} harmful content patterns")

        self.manipulation_patterns = {
            'gaslighting': r"(?i)(you're crazy|you're imagining|that never happened)",
            'social_engineering': r"(?i)(urgent|account.*suspend|verify.*account)",
            'persuasion': r"(?i)(limited time|act now|once in a lifetime)"
        }
        self.logger.debug(f"Loaded {len(self.manipulation_patterns)} manipulation patterns")

    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=500)
    def check_safety(self, text: str) -> Tuple[bool, List[SafetyViolation]]:
        """
        Check text for all safety violations.
        Returns (has_violations, list_of_violations)
        """
        self.logger.debug(f"Starting safety check for text: {text[:100]}...")
        violations = []

        # Check all pattern categories
        for category, check_func in [
            ("Adversarial", lambda: self._check_patterns(text, self.adversarial_patterns, "Adversarial")),
            ("PII", lambda: self._check_patterns(text, self.pii_patterns, "PII")),
            ("Harmful Content", lambda: self._check_patterns(text, self.harmful_content_patterns, "Harmful Content")),
            ("Manipulation", lambda: self._check_patterns(text, self.manipulation_patterns, "Manipulation"))
        ]:
            self.logger.debug(f"Checking {category} patterns")
            category_violations = check_func()
            if category_violations:
                self.logger.warning(f"Found {len(category_violations)} {category} violations")
            violations.extend(category_violations)

        # Additional custom checks
        token_violations = self._check_token_limit(text)
        rep_violations = self._check_repetition(text)
        violations.extend(token_violations + rep_violations)
        
        has_violations = len(violations) > 0
        if has_violations:
            self._log_violations(text, violations)
            
        self.logger.debug(f"Safety check completed. Found {len(violations)} violations")
        return has_violations, violations

    @DebugUtils.trace_function
    def _check_patterns(self, text: str, patterns: dict, category: str) -> List[SafetyViolation]:
        """Check text against a dictionary of patterns"""
        violations = []
        for pattern_name, pattern in patterns.items():
            self.logger.debug(f"Checking {category}/{pattern_name} pattern")
            if re.search(pattern, text):
                self.logger.warning(f"Pattern match found: {category}/{pattern_name}")
                violations.append(SafetyViolation(
                    type=f"{category}/{pattern_name}",
                    description=f"Detected potential {pattern_name} in text",
                    detected_pattern=pattern
                ))
        return violations

    @DebugUtils.trace_function
    def _check_token_limit(self, text: str, max_tokens: int = 1000) -> List[SafetyViolation]:
        """Check if text exceeds token limit (approximate)"""
        token_count = len(text.split())
        self.logger.debug(f"Checking token limit. Count: {token_count}, Max: {max_tokens}")
        
        if token_count > max_tokens:
            self.logger.warning(f"Token limit exceeded: {token_count} > {max_tokens}")
            return [SafetyViolation(
                type="Token Limit",
                description=f"Text exceeds maximum token limit of {max_tokens}",
                detected_pattern="token_count > max_tokens"
            )]
        return []

    @DebugUtils.trace_function
    def _check_repetition(self, text: str, threshold: float = 0.3) -> List[SafetyViolation]:
        """Check for suspicious repetition in text"""
        words = text.lower().split()
        if len(words) == 0:
            self.logger.debug("Empty text, skipping repetition check")
            return []
            
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        max_freq = max(word_freq.values())
        repetition_ratio = max_freq / len(words)
        self.logger.debug(f"Repetition check - Max frequency: {max_freq}, Ratio: {repetition_ratio:.2f}")
        
        if repetition_ratio > threshold:
            self.logger.warning(f"Suspicious repetition detected: {repetition_ratio:.2f} > {threshold}")
            return [SafetyViolation(
                type="Repetition",
                description="Detected suspicious repetition in text",
                detected_pattern=f"word_frequency > {threshold}"
            )]
        return []

    @DebugUtils.trace_function
    def _log_violations(self, text: str, violations: List[SafetyViolation]):
        """Log safety violations"""
        self.logger.warning(f"Found {len(violations)} safety violations")
        for violation in violations:
            self.logger.warning(
                f"Safety violation detected:\n"
                f"Type: {violation.type}\n"
                f"Description: {violation.description}\n"
                f"Pattern: {violation.detected_pattern}\n"
                f"Text snippet: {text[:100]}...\n"
            )
# Usage example:
"""
safety_framework = SafetyFramework()
text = "Here is my email: test@example.com and SSN: 123-45-6789"
has_violations, violations = safety_framework.check_safety(text)

if has_violations:
    print("Safety violations detected:")
    for violation in violations:
        print(f"- {violation.type}: {violation.description}")
"""
