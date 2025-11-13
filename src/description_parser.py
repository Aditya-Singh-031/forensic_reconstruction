"""
Forensic Description Parser
Extracts structured attributes from witness descriptions
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ForensicDescriptionParser:
    """Parse witness descriptions and extract structured forensic attributes."""
    
    def __init__(self, lexicon_path: Optional[str] = None):
        """
        Initialize the parser.
        
        Args:
            lexicon_path: Path to JSON lexicon file (optional)
        """
        logger.info("Initializing ForensicDescriptionParser...")
        
        # Load lexicon if provided
        self.lexicon = self._load_lexicon(lexicon_path) if lexicon_path else {}
        
        # Define attribute patterns
        self._setup_patterns()
        
        logger.info("✓ ForensicDescriptionParser initialized")
    
    def _load_lexicon(self, path: str) -> Dict:
        """Load attribute lexicon from JSON."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load lexicon: {e}")
            return {}
    
    def _setup_patterns(self):
        """Setup regex patterns for attribute extraction."""
        
        # Age patterns
        self.age_patterns = [
            r'(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*(?:years?|yrs?)?',  # 30-40 years
            r'(\d{1,2})\+?\s*(?:years?|yrs?)',  # 45 years, 45+
            r'(?:age|aged)\s*(\d{1,2})',  # age 30
        ]
        
        # Gender
        self.gender_keywords = {
            'male': ['male', 'man', 'boy', 'gentleman'],
            'female': ['female', 'woman', 'girl', 'lady']
        }
        
        # Complexion/Ethnicity
        self.complexion_keywords = {
            'indian': ['indian', 'south asian', 'desi'],
            'fair': ['fair', 'light', 'pale', 'white'],
            'dark': ['dark', 'dusky', 'brown'],
            'wheatish': ['wheatish', 'wheat', 'medium'],
            'african': ['african', 'black'],
            'asian': ['asian', 'east asian', 'chinese', 'japanese'],
            'caucasian': ['caucasian', 'white', 'european']
        }
        
        # Hair
        self.hair_color_keywords = ['black', 'brown', 'blonde', 'red', 'gray', 'grey', 'white']
        self.hair_length_keywords = ['long', 'short', 'medium', 'bald', 'balding']
        self.hair_style_keywords = ['straight', 'curly', 'wavy', 'spiky']
        
        # Facial hair
        self.facial_hair_keywords = {
            'mustache': ['mustache', 'moustache', 'stache'],
            'beard': ['beard', 'bearded'],
            'goatee': ['goatee'],
            'clean_shaven': ['clean shaven', 'clean-shaven', 'no beard']
        }
        
        # Eyes
        self.eye_color_keywords = ['blue', 'brown', 'green', 'hazel', 'gray', 'black']
        self.eye_size_keywords = ['large', 'small', 'big', 'wide', 'narrow']
        
        # Nose
        self.nose_keywords = ['large', 'small', 'pointed', 'flat', 'broad', 'narrow']
        
        # Distinctive features
        self.feature_keywords = {
            'glasses': ['glasses', 'spectacles', 'specs'],
            'scar': ['scar', 'scarred'],
            'tattoo': ['tattoo', 'tattooed'],
            'birthmark': ['birthmark', 'mole'],
            'earrings': ['earrings', 'piercing']
        }
        
        # Expression
        self.expression_keywords = ['angry', 'sad', 'happy', 'neutral', 'serious', 'smiling']
        
        # Build
        self.build_keywords = {
            'thin': ['thin', 'slim', 'lean', 'skinny'],
            'average': ['average', 'medium', 'normal'],
            'heavy': ['heavy', 'fat', 'overweight', 'obese'],
            'muscular': ['muscular', 'athletic', 'fit', 'strong']
        }
    
    def parse(self, description: str) -> Dict[str, Any]:
        """
        Parse a witness description and extract attributes.
        
        Args:
            description: Text description from witness
            
        Returns:
            Dictionary of extracted attributes with confidence scores
        """
        logger.info(f"Parsing description: {description[:50]}...")
        
        desc_lower = description.lower()
        
        attributes = {
            'age': self._extract_age(desc_lower),
            'gender': self._extract_gender(desc_lower),
            'complexion': self._extract_complexion(desc_lower),
            'hair': self._extract_hair(desc_lower),
            'facial_hair': self._extract_facial_hair(desc_lower),
            'eyes': self._extract_eyes(desc_lower),
            'nose': self._extract_nose(desc_lower),
            'distinctive_features': self._extract_features(desc_lower),
            'expression': self._extract_expression(desc_lower),
            'build': self._extract_build(desc_lower)
        }
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(attributes)
        
        result = {
            'raw_description': description,
            'attributes': attributes,
            'overall_confidence': confidence
        }
        
        logger.info(f"✓ Parsing complete (confidence: {confidence:.2f})")
        
        return result
    
    def _extract_age(self, text: str) -> Dict:
        """Extract age or age range."""
        for pattern in self.age_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 2:  # Range
                    return {
                        'value': f"{groups[0]}-{groups[1]}",
                        'min': int(groups[0]),
                        'max': int(groups[1]),
                        'confidence': 0.9
                    }
                else:  # Single age
                    age = int(groups[0])
                    return {
                        'value': str(age),
                        'min': age,
                        'max': age,
                        'confidence': 0.85
                    }
        return {'value': None, 'confidence': 0.0}
    
    def _extract_gender(self, text: str) -> Dict:
        """Extract gender."""
        for gender, keywords in self.gender_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return {'value': gender, 'confidence': 0.95}
        return {'value': None, 'confidence': 0.0}
    
    def _extract_complexion(self, text: str) -> Dict:
        """Extract complexion/ethnicity."""
        for complexion, keywords in self.complexion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return {'value': complexion, 'confidence': 0.8}
        return {'value': None, 'confidence': 0.0}
    
    def _extract_hair(self, text: str) -> Dict:
        """Extract hair attributes."""
        hair_attrs = {}
        
        # Color
        for color in self.hair_color_keywords:
            if color in text:
                hair_attrs['color'] = color
                break
        
        # Length
        for length in self.hair_length_keywords:
            if length in text:
                hair_attrs['length'] = length
                break
        
        # Style
        for style in self.hair_style_keywords:
            if style in text:
                hair_attrs['style'] = style
                break
        
        confidence = 0.7 if hair_attrs else 0.0
        return {'value': hair_attrs if hair_attrs else None, 'confidence': confidence}
    
    def _extract_facial_hair(self, text: str) -> Dict:
        """Extract facial hair attributes."""
        detected = []
        
        for fhair_type, keywords in self.facial_hair_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    detected.append(fhair_type)
                    break
        
        # Check for descriptors
        descriptors = []
        if 'thick' in text:
            descriptors.append('thick')
        if 'thin' in text:
            descriptors.append('thin')
        if 'bushy' in text:
            descriptors.append('bushy')
        
        result = {}
        if detected:
            result['type'] = detected
        if descriptors:
            result['descriptors'] = descriptors
        
        confidence = 0.8 if result else 0.0
        return {'value': result if result else None, 'confidence': confidence}
    
    def _extract_eyes(self, text: str) -> Dict:
        """Extract eye attributes."""
        eye_attrs = {}
        
        # Color
        for color in self.eye_color_keywords:
            if color in text and 'eye' in text:
                eye_attrs['color'] = color
                break
        
        # Size
        for size in self.eye_size_keywords:
            if size in text and 'eye' in text:
                eye_attrs['size'] = size
                break
        
        confidence = 0.7 if eye_attrs else 0.0
        return {'value': eye_attrs if eye_attrs else None, 'confidence': confidence}
    
    def _extract_nose(self, text: str) -> Dict:
        """Extract nose attributes."""
        for keyword in self.nose_keywords:
            if keyword in text and 'nose' in text:
                return {'value': keyword, 'confidence': 0.7}
        return {'value': None, 'confidence': 0.0}
    
    def _extract_features(self, text: str) -> Dict:
        """Extract distinctive features."""
        features = []
        
        for feature, keywords in self.feature_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    features.append(feature)
                    break
        
        confidence = 0.75 if features else 0.0
        return {'value': features if features else None, 'confidence': confidence}
    
    def _extract_expression(self, text: str) -> Dict:
        """Extract facial expression."""
        for expr in self.expression_keywords:
            if expr in text:
                return {'value': expr, 'confidence': 0.65}
        return {'value': None, 'confidence': 0.0}
    
    def _extract_build(self, text: str) -> Dict:
        """Extract body build."""
        for build, keywords in self.build_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return {'value': build, 'confidence': 0.7}
        return {'value': None, 'confidence': 0.0}
    
    def _calculate_confidence(self, attributes: Dict) -> float:
        """Calculate overall confidence score."""
        confidences = []
        for attr in attributes.values():
            if isinstance(attr, dict) and 'confidence' in attr:
                confidences.append(attr['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
