
import streamlit as st
import requests
import json
import pandas as pd
import google.generativeai as genai
from typing import Dict, Any, Optional, List
import re
from dataclasses import dataclass

# Hardcoded API credentials
APPLICATION_ID = '1aad9geoo60y5e8z0jr496fwcq1ybti4'
SHARED_SECRET = 'jHlILAQ-jU22FwsWtfZ1zQ'
REFRESH_TOKEN = 'mKJx-l2zTVtY8YFQ38oN6tpMIPTdBgxqql191I1KjzI'
GEMINI_API_KEY = 'AIzaSyChNGDaSBBkye08bY0-3s9stjukCIvIKhg'

# Configure page
st.set_page_config(
    page_title="Gemstone Bill of Materials Calculator",
    page_icon="💎",
    layout="wide"
)

@dataclass
class ProductAttributes:
    """Data class to hold all product attributes"""
    id: str = ""
    title: str = ""
    description: str = ""
    pick_1: str = ""
    pick_2: str = ""
    pick_3: str = ""
    material: str = ""
    metal_purity: str = ""
    exact_carat_total_weight: str = ""
    right_weight: str = ""
    width: str = ""
    weight: str = ""
    size: str = ""
    number_of_diamonds: str = ""
    number_of_gemstones: str = ""
    metal: str = ""
    main_stone: str = ""
    gender: str = ""
    casting_weight: str = ""
    gemstone_carat: str = ""
    diamond_carat: str = ""
    stone_type: str = ""
    stone_shape: str = ""
    stone_dimensions: str = ""
    stone_carat: str = ""
    stone_quantity: str = ""
    supplier_name: str = ""
    supplier_code: str = ""
    mfg_part_number: str = ""
    part_number_2: str = ""

@dataclass
class GemstoneDetail:
    """Data class for individual gemstone details"""
    gemstone_type: str = ""
    gemstone_shape: str = ""
    gemstone_size: str = ""
    quantity: int = 1
    gemstone_carat: float = 0.0
    carat_per_qty: float = 0.0
    product_id: str = ""

class ChannelAdvisorAPI:
    """Handle Channel Advisor API interactions"""
    
    def __init__(self, base_url: str, api_key: str = None, developer_key: str = None, 
                 application_id: str = None, shared_secret: str = None, refresh_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.developer_key = developer_key
        self.application_id = application_id
        self.shared_secret = shared_secret
        self.refresh_token = refresh_token
        self.access_token = None
        
        self.headers = {
            'Content-Type': 'application/json',
        }
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def refresh_access_token(self) -> str:
        """Refresh the access token using OAuth credentials"""
        if not all([self.application_id, self.shared_secret, self.refresh_token]):
            raise Exception("Missing OAuth credentials for token refresh")
        
        import base64
        auth_string = base64.b64encode(f"{self.application_id}:{self.shared_secret}".encode()).decode()
        
        token_url = f"{self.base_url}/oauth2/token"
        headers = {
            'Authorization': f'Basic {auth_string}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token
        }
        
        response = requests.post(token_url, headers=headers, data=data)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            self.headers['Authorization'] = f'Bearer {self.access_token}'
            return self.access_token
        else:
            raise Exception(f"Token refresh failed: {response.status_code} - {response.text}")
    
    def make_api_request(self, endpoint: str, method: str = 'GET', payload: dict = None) -> Optional[dict]:
        """Make API request with automatic token refresh"""
        if not self.access_token and self.refresh_token:
            self.refresh_access_token()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=payload)
            elif method.upper() == 'PATCH':
                response = requests.patch(url, headers=self.headers, json=payload)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle 401 with token refresh
            if response.status_code == 401 and self.refresh_token:
                self.refresh_access_token()
                # Retry the request
                if method.upper() == 'GET':
                    response = requests.get(url, headers=self.headers)
                elif method.upper() == 'POST':
                    response = requests.post(url, headers=self.headers, json=payload)
                elif method.upper() == 'PATCH':
                    response = requests.patch(url, headers=self.headers, json=payload)
            
            if response.status_code >= 200 and response.status_code < 300:
                return response.json() if response.text else {"success": True}
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Network error: {str(e)}")
            return None
    
    def get_product_by_sku(self, sku: str) -> Optional[Dict[str, Any]]:
        """Get product information by SKU"""
        endpoint = f"/v1/Products?$filter=Sku eq '{sku}'"
        response = self.make_api_request(endpoint)
        
        if response and response.get('value') and len(response['value']) > 0:
            product_id = response['value'][0]['ID']
            return self.fetch_product_details(product_id)
        return None
    
    def fetch_product_details(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed product information by ID"""
        endpoints = {
            'basic': f'/v1/Products({product_id})',
            'attributes': f'/v1/Products({product_id})/Attributes',
            'labels': f'/v1/Products({product_id})/Labels'
        }
        
        details = {'ID': product_id}
        
        for key, endpoint in endpoints.items():
            response = self.make_api_request(endpoint)
            if response:
                details[key] = response
            else:
                st.warning(f"Failed to fetch {key} details")
        
        return details if len(details) > 1 else None

class ImprovedGeminiProcessor:
    """Improved Gemini AI processing for gemstone data extraction"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def extract_all_attributes_and_gemstones(self, raw_data: Dict[str, Any]) -> tuple[ProductAttributes, List[GemstoneDetail]]:
        """Extract ALL Channel Advisor attributes and then process gemstone details"""
        
        # Initialize attributes object
        attributes = ProductAttributes()
        
        # Extract basic information
        if 'basic' in raw_data:
            basic_data = raw_data['basic']
            attributes.id = str(basic_data.get('ID', ''))
            attributes.title = basic_data.get('Title', '')
            attributes.description = basic_data.get('Description', '')
        
        # Parse ALL attributes data
        if 'attributes' in raw_data and 'value' in raw_data['attributes']:
            attributes_data = raw_data['attributes']['value']
            
            # Create a dictionary for easier lookup - normalize to lowercase
            attr_dict = {}
            for attr in attributes_data:
                name = attr.get('Name', '').lower().strip()
                value = str(attr.get('Value', '')).strip()
                attr_dict[name] = value
            
            # Map ALL the attributes from your example
            attributes.pick_1 = attr_dict.get('pick 1', '')
            attributes.pick_2 = attr_dict.get('pick 2', '')
            attributes.pick_3 = attr_dict.get('pick 3', '')
            attributes.material = attr_dict.get('material', '')
            attributes.metal_purity = attr_dict.get('metal purity', '')
            attributes.exact_carat_total_weight = attr_dict.get('exact carat total weight', '')
            attributes.right_weight = attr_dict.get('right weight', '')
            attributes.width = attr_dict.get('width', '')
            attributes.weight = attr_dict.get('weight', '')
            attributes.size = attr_dict.get('size', '')
            attributes.number_of_diamonds = attr_dict.get('number of diamonds', '')
            attributes.number_of_gemstones = attr_dict.get('number of gemstones', '')
            attributes.metal = attr_dict.get('metal', '')
            attributes.main_stone = attr_dict.get('main stone', '')
            attributes.gender = attr_dict.get('gender', '')
            attributes.casting_weight = attr_dict.get('casting weight', '')
            attributes.diamond_carat = attr_dict.get('diamond carat', '')
            attributes.stone_type = attr_dict.get('stone type', '')
            attributes.stone_shape = attr_dict.get('stone shape', '')
            attributes.stone_dimensions = attr_dict.get('stone dimensions', '')
            attributes.stone_carat = attr_dict.get('stone carat', '')
            attributes.stone_quantity = attr_dict.get('stone quantity', '')
            
            # Additional attributes that might exist
            attributes.supplier_name = attr_dict.get('supplier name', '')
            attributes.supplier_code = attr_dict.get('supplier code', '')
            attributes.mfg_part_number = attr_dict.get('mfg part number', '')
            attributes.part_number_2 = attr_dict.get('part number 2', '')
        
        # Now extract gemstone details using the complete attribute data
        gemstone_details = self._extract_gemstone_details(attributes, raw_data)
        
        return attributes, gemstone_details
    
    def _extract_gemstone_details(self, attributes: ProductAttributes, raw_data: Dict[str, Any]) -> List[GemstoneDetail]:
        """Extract gemstone details from the complete attribute data"""
        
        gemstone_details = []
        
        # Method 1: Direct extraction from attributes
        if attributes.stone_type and attributes.stone_type.lower() not in ['diamond', 'n/a', 'not available', '']:
            gemstone = GemstoneDetail()
            gemstone.gemstone_type = attributes.stone_type
            gemstone.gemstone_shape = attributes.stone_shape
            
            # For size, try multiple sources
            size_sources = [
                attributes.stone_dimensions,
                attributes.pick_2,  # Often contains size info like "10x8 oval center"
                attributes.size
            ]
            
            for size_source in size_sources:
                if size_source and size_source.lower() not in ['n/a', 'not available', '']:
                    # Extract size from pick_2 if it contains dimensions
                    size_match = re.search(r'(\d+x\d+|\d+\.\d+)\s*(?:mm)?', size_source.lower())
                    if size_match:
                        gemstone.gemstone_size = size_match.group(1) + ' mm'
                        break
                    elif 'mm' in size_source.lower():
                        gemstone.gemstone_size = size_source
                        break
            
            # Parse quantity
            try:
                gemstone.quantity = int(attributes.stone_quantity) if attributes.stone_quantity else 1
            except:
                gemstone.quantity = 1
            
            # Parse carat
            try:
                carat_value = attributes.stone_carat
                if carat_value:
                    # Remove any non-numeric characters except decimal point
                    carat_clean = re.sub(r'[^\d.]', '', carat_value)
                    gemstone.gemstone_carat = float(carat_clean) if carat_clean else 0.0
                else:
                    gemstone.gemstone_carat = 0.0
            except:
                gemstone.gemstone_carat = 0.0
            
            gemstone_details.append(gemstone)
        
        # Method 2: Extract from Pick fields if no direct stone data
        if not gemstone_details:
            gemstone_details = self._extract_from_pick_fields(attributes)
        
        # Method 3: Gemini AI fallback
        if not gemstone_details:
            gemstone_details = self._gemini_fallback_extraction(raw_data, attributes)
        
        return gemstone_details
    
    def _extract_from_pick_fields(self, attributes: ProductAttributes) -> List[GemstoneDetail]:
        """Extract gemstone info from Pick fields"""
        
        gemstone_details = []
        
        # Analyze Pick 2 which often contains center stone info
        if attributes.pick_2:
            pick2_lower = attributes.pick_2.lower()
            
            # Look for size pattern
            size_match = re.search(r'(\d+x\d+)', pick2_lower)
            
            # Look for shape
            shapes = ['oval', 'round', 'emerald', 'pear', 'marquise', 'cushion', 'princess', 'asscher']
            detected_shape = None
            for shape in shapes:
                if shape in pick2_lower:
                    detected_shape = shape.title()
                    break
            
            # If we found size info, create gemstone detail
            if size_match or detected_shape:
                gemstone = GemstoneDetail()
                
                # Get gemstone type from stone_type attribute or title
                if attributes.stone_type and attributes.stone_type.lower() not in ['diamond', 'n/a']:
                    gemstone.gemstone_type = attributes.stone_type
                else:
                    # Try to extract from title
                    title_lower = attributes.title.lower()
                    gemstone_types = ['morganite', 'sapphire', 'ruby', 'emerald', 'topaz', 'amethyst', 'garnet', 'peridot', 'citrine']
                    for gem_type in gemstone_types:
                        if gem_type in title_lower:
                            gemstone.gemstone_type = gem_type.title()
                            break
                
                if size_match:
                    gemstone.gemstone_size = size_match.group(1) + ' mm'
                
                if detected_shape:
                    gemstone.gemstone_shape = detected_shape
                
                # Get carat from stone_carat attribute
                try:
                    if attributes.stone_carat:
                        gemstone.gemstone_carat = float(attributes.stone_carat)
                except:
                    gemstone.gemstone_carat = 0.0
                
                gemstone.quantity = 1  # Assume 1 center stone
                
                if gemstone.gemstone_type:  # Only add if we found a gemstone type
                    gemstone_details.append(gemstone)
        
        return gemstone_details
    
    def _gemini_fallback_extraction(self, raw_data: Dict[str, Any], attributes: ProductAttributes) -> List[GemstoneDetail]:
        """Fallback to Gemini processing with improved prompts"""
        
        prompt = f"""
        Analyze this jewelry product data and extract ONLY gemstone information (exclude diamonds):

        Product Title: {attributes.title}
        Product Description: {attributes.description}
        Pick 1: {attributes.pick_1}
        Pick 2: {attributes.pick_2}
        Pick 3: {attributes.pick_3}
        Stone Type: {attributes.stone_type}
        Stone Shape: {attributes.stone_shape}
        Stone Carat: {attributes.stone_carat}
        Stone Quantity: {attributes.stone_quantity}

        INSTRUCTIONS:
        1. Focus on non-diamond gemstones (morganite, sapphire, ruby, emerald, topaz, etc.)
        2. Extract size information from Pick 2 field or other sources
        3. Use the Stone Type, Stone Shape, and Stone Carat attributes

        Return ONLY a JSON array with this exact structure for each gemstone:
        [
            {{
                "gemstone_type": "exact gemstone name",
                "gemstone_shape": "shape (round, oval, emerald, etc.)",
                "gemstone_size": "size with units (e.g., 10x8 mm, 5.0 mm)",
                "quantity": number_as_integer,
                "gemstone_carat": number_as_float
            }}
        ]

        Return ONLY the JSON array, no other text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up the response to extract JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            # Parse JSON response
            gemstone_data = json.loads(response_text)
            
            gemstone_details = []
            for gem_data in gemstone_data:
                gemstone = GemstoneDetail()
                gemstone.gemstone_type = gem_data.get('gemstone_type', '')
                gemstone.gemstone_shape = gem_data.get('gemstone_shape', '')
                gemstone.gemstone_size = gem_data.get('gemstone_size', '')
                gemstone.quantity = gem_data.get('quantity', 1)
                gemstone.gemstone_carat = gem_data.get('gemstone_carat', 0.0)
                gemstone_details.append(gemstone)
            
            return gemstone_details
            
        except Exception as e:
            st.warning(f"Gemini processing failed: {str(e)}")
            return []

def generate_comprehensive_summary(attributes: ProductAttributes, gemstone_details: List[GemstoneDetail]) -> str:
    """Generate a comprehensive summary of the jewelry product"""
    
    # Basic product type detection
    product_type = "jewelry piece"
    title_lower = attributes.title.lower()
    if "ring" in title_lower:
        if "engagement" in title_lower:
            product_type = "engagement ring"
        elif "wedding" in title_lower:
            product_type = "wedding ring"
        else:
            product_type = "ring"
    elif "necklace" in title_lower:
        product_type = "necklace"
    elif "earring" in title_lower:
        product_type = "earrings"
    elif "bracelet" in title_lower:
        product_type = "bracelet"
    elif "pendant" in title_lower:
        product_type = "pendant"
    
    # Gender
    gender = attributes.gender.lower() if attributes.gender else ""
    gender_text = ""
    if "female" in gender or "women" in gender:
        gender_text = "women's "
    elif "male" in gender or "men" in gender:
        gender_text = "men's "
    
    # Metal information
    metal_info = ""
    if attributes.metal or attributes.material:
        metal = attributes.metal or attributes.material
        if attributes.metal_purity:
            metal_info = f"{attributes.metal_purity} {metal.lower()}"
        else:
            metal_info = metal.lower()
    
    # Setting type detection
    setting_type = ""
    if attributes.number_of_diamonds and int(attributes.number_of_diamonds or 0) > 0:
        if "halo" in title_lower:
            setting_type = "halo setting"
        else:
            setting_type = "diamond-accented setting"
    
    # Main gemstone information
    main_gemstone_info = ""
    if gemstone_details:
        main_gem = gemstone_details[0]  # Assume first is main stone
        
        # Determine if it's treated
        treated_text = ""
        if "treated" in title_lower or "treated" in attributes.description.lower():
            treated_text = "treated "
        
        # Build gemstone description
        shape_text = f", brilliant-cut" if "brilliant" in (main_gem.gemstone_shape or "").lower() else f"-cut" if main_gem.gemstone_shape else ""
        carat_text = f" weighing {main_gem.gemstone_carat} carats" if main_gem.gemstone_carat > 0 else ""
        size_text = f", measuring {main_gem.gemstone_size} in diameter" if main_gem.gemstone_size else ""
        
        main_gemstone_info = f"{treated_text}{main_gem.gemstone_type.lower()}{shape_text}{size_text}{carat_text}"
    
    # Diamond information
    diamond_info = ""
    if attributes.number_of_diamonds and int(attributes.number_of_diamonds or 0) > 0:
        num_diamonds = attributes.number_of_diamonds
        if setting_type == "halo setting":
            diamond_info = f" It is accompanied by {num_diamonds} accent diamonds, arranged in a halo setting around the central stone, enhancing its brilliance and overall appeal."
        else:
            diamond_info = f" It is accented by {num_diamonds} diamonds."
        
        # Add total carat weight if available
        if attributes.exact_carat_total_weight:
            diamond_info += f" The total carat weight for the entire {product_type} is approximately {attributes.exact_carat_total_weight} carats."
    
    # Physical specifications
    physical_specs = ""
    specs = []
    if attributes.width:
        specs.append(f"width of {attributes.width} mm")
    if attributes.casting_weight:
        specs.append(f"casting weight of {attributes.casting_weight} grams")
    if attributes.size:
        specs.append(f"available in size {attributes.size}")
    
    if specs:
        physical_specs = f" The band is made of {metal_info} with a {', '.join(specs)}."
    elif metal_info:
        physical_specs = f" The piece is crafted in {metal_info}."
    
    # Handle discrepancies (like Imperial Topaz vs Blue Topaz)
    discrepancy_note = ""
    if gemstone_details and attributes.stone_type:
        extracted_type = gemstone_details[0].gemstone_type.lower()
        attribute_type = attributes.stone_type.lower()
        
        # Check for common discrepancies
        if "imperial" in attribute_type and "blue" in title_lower and "topaz" in attribute_type:
            pick_2_lower = attributes.pick_2.lower()
            if "blue" in pick_2_lower:
                discrepancy_note = f' Some fields mention "{attributes.stone_type}," but based on context and pick fields, the correct main gemstone appears to be treated blue topaz.'
    
    # Build final summary
    summary = f"This {product_type} is a {gender_text}"
    
    # Add specific product details
    if setting_type:
        summary += f"{product_type} crafted in {metal_info}, designed with a central {main_gemstone_info} and accented by diamonds."
    else:
        summary += f"{product_type} featuring {main_gemstone_info}."
    
    # Add main gemstone details
    if gemstone_details:
        main_gem = gemstone_details[0]
        summary += f" The main gemstone is a {main_gem.gemstone_shape.lower() if main_gem.gemstone_shape else 'cut'} {main_gemstone_info}."
    
    # Add diamond information
    summary += diamond_info
    
    # Add physical specifications
    summary += physical_specs
    
    # Add gender and sizing if relevant
    if gender_text and attributes.size:
        summary += f" This elegant piece is designed for {gender_text.strip()} and is available in size {attributes.size}."
    elif gender_text:
        summary += f" This elegant piece is designed for {gender_text.strip()}."
    
    # Add discrepancy note if any
    summary += discrepancy_note
    
    return summary

# Updated processing function
def process_with_complete_extraction(raw_product_data, gemini_api_key):
    """Extract all attributes and gemstone details"""
    
    # Initialize improved processor
    improved_processor = ImprovedGeminiProcessor(gemini_api_key)
    
    # Extract all attributes and gemstone details
    attributes, gemstone_details = improved_processor.extract_all_attributes_and_gemstones(raw_product_data)
    
    # Generate comprehensive summary
    comprehensive_summary = generate_comprehensive_summary(attributes, gemstone_details)
    
    # Create comprehensive summary
    structured_info = f"""
COMPLETE PRODUCT ANALYSIS:

Basic Information:
- Product ID: {attributes.id}
- Title: {attributes.title}
- Description: {attributes.description}

Pick Fields:
- Pick 1: {attributes.pick_1}
- Pick 2: {attributes.pick_2}
- Pick 3: {attributes.pick_3}

Material & Metal:
- Material: {attributes.material}
- Metal: {attributes.metal}
- Metal Purity: {attributes.metal_purity}

Measurements:
- Weight: {attributes.weight}
- Width: {attributes.width}
- Size: {attributes.size}
- Casting Weight: {attributes.casting_weight}

Stone Information:
- Stone Type: {attributes.stone_type}
- Stone Shape: {attributes.stone_shape}
- Stone Dimensions: {attributes.stone_dimensions}
- Stone Carat: {attributes.stone_carat}
- Stone Quantity: {attributes.stone_quantity}

Diamond Information:
- Number of Diamonds: {attributes.number_of_diamonds}
- Diamond Carat: {attributes.diamond_carat}

Other:
- Main Stone: {attributes.main_stone}
- Gender: {attributes.gender}
- Exact Carat Total Weight: {attributes.exact_carat_total_weight}

EXTRACTED GEMSTONE DETAILS:
"""
    
    for i, gem in enumerate(gemstone_details, 1):
        structured_info += f"""
Gemstone {i}:
- Type: {gem.gemstone_type}
- Shape: {gem.gemstone_shape}
- Size: {gem.gemstone_size}
- Quantity: {gem.quantity}
- Carat: {gem.gemstone_carat}
"""
    
    if not gemstone_details:
        structured_info += "No gemstones detected in this product."
    
    return attributes, gemstone_details, structured_info, comprehensive_summary

class GemstonePriceCalculator:
    """Handle gemstone price calculations using the gemstone price data"""
    
    def __init__(self, round_gemstone_prices: pd.DataFrame, nonround_gemstone_prices: pd.DataFrame):
        self.round_prices = round_gemstone_prices
        self.nonround_prices = nonround_gemstone_prices
    
    def calculate_gemstone_bill(self, gemstone_details: List[GemstoneDetail]) -> Dict[str, Any]:
        """Calculate costs based on extracted gemstone details and price data"""
        
        if not gemstone_details:
            return {
                'gemstones': [],
                'total_bill': 0.0,
                'error': 'No gemstone details provided'
            }
        
        bill_details = []
        total_bill = 0.0
        
        for i, gem in enumerate(gemstone_details):
            st.info(f"Processing gemstone #{i+1}: {gem.gemstone_type} - {gem.gemstone_shape} - {gem.gemstone_size}")
            
            # Normalize values for matching
            shape = (gem.gemstone_shape or "").strip()
            size = (gem.gemstone_size or "").strip().replace(" mm", "").replace("mm", "")
            gemstone_type = (gem.gemstone_type or "").strip()
            quantity = gem.quantity or 1
            
            # Debug output
            #st.write(f"Debug - Looking for: Shape='{shape}', Size='{size}', Type='{gemstone_type}'")
            
            # Determine which dataset to use based on shape
            if shape.upper() == "ROUND":
                dataset = self.round_prices
                #st.write("Using round gemstone dataset")
            else:
                dataset = self.nonround_prices  
                #st.write("Using non-round gemstone dataset")
            
            # Find matching price with improved logic
            match = None
            best_match_score = 0
            best_match_details = ""
            
            if not dataset.empty:
                #st.write(f"Dataset has {len(dataset)} rows")
                
                # Display available data for debugging (show relevant matches)
                #st.write("Sample available gemstones in dataset:")
                relevant_samples = dataset[dataset['gemstone'].str.contains(gemstone_type, case=False, na=False)].head(3)
                if relevant_samples.empty:
                    relevant_samples = dataset.head(5)
                
                for idx, row in relevant_samples.iterrows():
                    st.write(f"  - {row.get('gemstone', 'N/A')} | {row.get('shape', 'N/A')} | {row.get('size', 'N/A')} | ${row.get('itemPrice', 0)}")
                
                for idx, row in dataset.iterrows():
                    row_shape = str(row.get('shape', '')).strip()
                    row_size = str(row.get('size', '')).strip()
                    row_gemstone = str(row.get('gemstone', '')).strip()
                    
                    # Calculate match score
                    match_score = 0
                    match_details = []
                    
                    # Gemstone type matching (most important) - case insensitive
                    if self._gemstone_types_match(gemstone_type.lower(), row_gemstone.lower()):
                        match_score += 3
                        match_details.append("gemstone_match")
                    
                    # Shape matching - case insensitive
                    if shape and row_shape:
                        if shape.upper() == row_shape.upper():
                            match_score += 2
                            match_details.append("exact_shape_match")
                        elif self._shapes_match(shape.upper(), row_shape.upper()):
                            match_score += 1.5
                            match_details.append("similar_shape_match")
                    elif not shape or not row_shape or row_shape == '-':  # If shape is missing, don't penalize
                        match_score += 1
                        match_details.append("shape_tolerance")
                    
                    # Size matching (flexible)
                    if self._sizes_match(size, row_size):
                        match_score += 2
                        match_details.append("size_match")
                    elif not size or not row_size or row_size == '-':  # If size is missing, don't penalize heavily
                        match_score += 0.5
                        match_details.append("size_tolerance")
                    
                    # Keep track of best match
                    if match_score > best_match_score:
                        best_match_score = match_score
                        match = row
                        best_match_details = " + ".join(match_details)
                        #st.write(f"New best match found: {row_gemstone} | {row_shape} | {row_size} | ${row.get('itemPrice', 0)} (Score: {match_score:.1f} - {best_match_details})")
            
            # Calculate pricing
            if match is not None and best_match_score >= 3:  # Require at least gemstone type match
                unit_price = float(match.get('itemPrice', 0))
                product_id = str(match.get('productId', ''))
                gem.product_id = product_id
                note = f"Matched PID: {product_id} (Score: {best_match_score:.1f} - {best_match_details})"
                st.success(f"✅ Found match: ${unit_price} per unit - {match.get('gemstone', '')} {match.get('shape', '')} {match.get('size', '')}")
            else:
                unit_price = 0.0
                note = f"No suitable match found (Best score: {best_match_score:.1f})"
                st.warning(f"❌ No match found for {gemstone_type} {shape} {size}")
                
                # Show what we were looking for vs what's available
                if best_match_score > 0:
                    st.write(f"Best partial match was: {match.get('gemstone', '')} {match.get('shape', '')} {match.get('size', '')} with score {best_match_score:.1f}")
            
            subtotal = unit_price * quantity
            total_bill += subtotal
            
            # Create bill entry
            bill_entry = {
                'gemstone_number': i + 1,
                'gemstoneType': gem.gemstone_type,
                'gemstoneShape': gem.gemstone_shape,
                'gemstoneSize': gem.gemstone_size,
                'quantity': quantity,
                'unit_price': round(unit_price, 2),
                'total_bill': round(subtotal, 2),
                'gemstoneCarat': gem.gemstone_carat,
                'productId': gem.product_id,
                'note': note
            }
            
            bill_details.append(bill_entry)
        
        return {
            'gemstones': bill_details,
            'total_bill': round(total_bill, 2)
        }
    
    def _gemstone_types_match(self, extracted_type: str, dataset_type: str) -> bool:
        """Check if gemstone types match with flexible matching"""
        if not extracted_type or not dataset_type:
            return False
        
        # Convert to lowercase for comparison
        extracted_type = extracted_type.lower().strip()
        dataset_type = dataset_type.lower().strip()
        
        # Direct match
        if extracted_type == dataset_type:
            return True
        
        # Handle common variations and your actual data
        type_aliases = {
            'citrine': ['citrine'],
            'blue sapphire': ['sapphire', 'blue sapphire', 'sapp', 'bsapp'],
            'sapphire': ['sapphire', 'blue sapphire', 'sapp', 'bsapp', 'pink sapphire', 'yellow sapphire', 'orange sapp'],
            'ruby': ['ruby', 'rby', 'red ruby'],
            'emerald': ['emerald', 'em'],
            'amethyst': ['amethyst', 'amth'],
            'topaz': ['topaz', 'topz', 'blue topaz', 'btopz', 'pink topaz', 'ptz'],
            'morganite': ['morganite', 'morg'],
            'garnet': ['garnet', 'garn'],
            'peridot': ['peridot', 'per'],
            'aquamarine': ['aquamarine', 'aqua'],
            'moissanite': ['moissanite', 'mois'],
            'opal': ['opal']
        }
        
        # Check if either type appears in the other's alias list
        for base_type, aliases in type_aliases.items():
            if extracted_type in aliases and dataset_type in aliases:
                return True
            if base_type == extracted_type and dataset_type in aliases:
                return True
            if base_type == dataset_type and extracted_type in aliases:
                return True
        
        # Handle specific cases from your data
        if 'sapphire' in extracted_type and 'sapphire' in dataset_type:
            return True
        if 'topaz' in extracted_type and 'topaz' in dataset_type:
            return True
        if 'ruby' in extracted_type and 'ruby' in dataset_type:
            return True
        
        # Partial matching for composite names
        extracted_words = extracted_type.split()
        dataset_words = dataset_type.split()
        
        # If any significant word matches
        for ext_word in extracted_words:
            if len(ext_word) > 3:  # Only consider significant words
                for dat_word in dataset_words:
                    if ext_word in dat_word or dat_word in ext_word:
                        return True
        
        return False
    
    def _shapes_match(self, extracted_shape: str, dataset_shape: str) -> bool:
        """Check if shapes match with flexible matching for similar shapes"""
        if not extracted_shape or not dataset_shape:
            return False
        
        # Handle common shape aliases
        shape_aliases = {
            'OVAL': ['OVAL', 'OV'],
            'ROUND': ['ROUND', 'R'],
            'EMERALD': ['EMERALD', 'EM'],
            'PEAR': ['PEAR', 'PS'],  # PS = Pear Shape
            'HEART': ['HEART', 'HT'],
            'CUSHION': ['CUSHION', 'CU'],
            'ASSCHER': ['ASSCHER', 'AS'],
            'SQUARE': ['SQUARE', 'SQ'],
            'BAGUETTE': ['BAGUETTE', 'STRAIGHT BAGUETTE', 'STB']
        }
        
        for base_shape, aliases in shape_aliases.items():
            if extracted_shape in aliases and dataset_shape in aliases:
                return True
        
        return False
    
    def _sizes_match(self, extracted_size: str, dataset_size: str) -> bool:
        """Check if sizes match with flexible matching"""
        if not extracted_size or not dataset_size:
            return False
        
        # Handle dash/missing values
        if dataset_size == '-' or extracted_size == '-':
            return False
        
        # Clean sizes (remove mm, spaces, etc.)
        clean_extracted = extracted_size.replace('mm', '').replace(' ', '').strip().lower()
        clean_dataset = dataset_size.replace('mm', '').replace(' ', '').strip().lower()
        
        # Direct match
        if clean_extracted == clean_dataset:
            return True
        
        # Handle case differences (10x8 vs 10X8)
        if clean_extracted.replace('x', 'X') == clean_dataset.replace('x', 'X'):
            return True
        
        # For dimensional sizes like "10x8", try to match dimensions regardless of order
        if 'x' in clean_extracted.lower() and 'x' in clean_dataset.lower():
            try:
                # Split and sort dimensions to handle 10x8 vs 8x10
                ext_dims = sorted([float(d) for d in clean_extracted.replace('x', 'X').split('X')])
                dat_dims = sorted([float(d) for d in clean_dataset.replace('x', 'X').split('X')])
                
                if len(ext_dims) == 2 and len(dat_dims) == 2:
                    # Allow small tolerance for floating point comparison
                    return (abs(ext_dims[0] - dat_dims[0]) < 0.1 and 
                           abs(ext_dims[1] - dat_dims[1]) < 0.1)
            except (ValueError, IndexError):
                pass
        
        # For single dimensions, try approximate matching
        try:
            if 'x' not in clean_extracted and 'x' not in clean_dataset:
                ext_val = float(clean_extracted)
                dat_val = float(clean_dataset)
                # Allow 0.1mm tolerance
                return abs(ext_val - dat_val) <= 0.1
        except ValueError:
            pass
        
        return False

def load_sample_gemstone_data():
    """Load sample gemstone pricing data - this should not be used when real CSV files are uploaded"""
    # This function creates minimal sample data for testing when no CSV files are provided
    # Since you have actual CSV files, this won't be used
    
    # Minimal round gemstone data for fallback
    round_data = {
        'category': ['Gemstone'] * 3,
        'productId': ['2.3/R/BSAPP', '3/R/RBY', '4.5/R/TOPAZ'],
        'carat': [0.75, 1.15, 11],
        'ppc': [50, 40, 40],
        'caratPerUnit': [0.25, 0.25, 0.25],
        'itemPrice': [12.5, 10, 10],
        'shape': ['ROUND', 'ROUND', 'ROUND'],
        'size': ['2.3', '3', '4.5'],
        'gemstone': ['blue sapphire', 'red ruby', 'topaz'],
        'description': ['2.3 ROUND Blue Sapphire', '3 ROUND Red Ruby', '4.5 ROUND Topaz']
    }
    
    # Minimal non-round gemstone data for fallback
    nonround_data = {
        'category': ['Gemstone'] * 3,
        'size': ['10x8', '8x6', '7x5'],
        'shape': ['oval', 'EMERALD', 'oval'],
        'gemstone': ['citrine', 'Blue Sapphire', 'ruby'],
        'carat': [2.68, 1.5, 0],
        'productId': ['10x8/OV/CIT', '8x6/EM/BSAPP', '7x5/OV/RBY'],
        'ppc': [32, 50, 40],
        'caratPerUnit': [0.25, 0.25, 0.25],
        'itemPrice': [8, 12.5, 10],
        'description': ['10x8 Oval Citrine', '8x6 Emerald Blue Sapphire', '7x5 Oval Ruby']
    }
    
    return pd.DataFrame(round_data), pd.DataFrame(nonround_data)

def main():
    st.title("💎 Gemstone Bill of Materials Calculator")
    st.markdown("Enter a SKU to get detailed gemstone product information and cost calculation")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Configuration (hardcoded but still showing)
        ca_endpoint = st.text_input(
            "Channel Advisor Base URL", 
            value="https://api.channeladvisor.com",
            help="Channel Advisor API base URL"
        )
        
        # Show that credentials are configured
        st.info("✅ API Credentials: Pre-configured")
        
        # File upload for gemstone price data
        st.subheader("Gemstone Price Data")
        round_file = st.file_uploader(
            "Upload Round Gemstone Prices",
            type=['csv', 'xlsx'],
            help="Upload round gemstone price dataset"
        )
        
        nonround_file = st.file_uploader(
            "Upload Non-Round Gemstone Prices", 
            type=['csv', 'xlsx'],
            help="Upload non-round gemstone price dataset"
        )
        
        use_sample_data = st.checkbox("Use Sample Data", value=True)
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input")
        sku_input = st.text_input(
            "Enter SKU", 
            placeholder="e.g., SKU12345",
            help="Enter the SKU to look up gemstone product information"
        )
        
        process_button = st.button("Process SKU", type="primary")
    
    with col2:
        st.subheader("Process Steps")
        step1_status = st.empty()
        step2_status = st.empty() 
        step3_status = st.empty()
    
    # Process SKU when button is clicked
    if process_button and sku_input and ca_endpoint:
        
        # Initialize Channel Advisor API with hardcoded credentials
        ca_api = ChannelAdvisorAPI(
            base_url=ca_endpoint,
            application_id=APPLICATION_ID,
            shared_secret=SHARED_SECRET,
            refresh_token=REFRESH_TOKEN
        )
        
        # Load gemstone price data
        if use_sample_data:
            round_df, nonround_df = load_sample_gemstone_data()
        else:
            round_df = pd.DataFrame()
            nonround_df = pd.DataFrame()
            
            if round_file:
                try:
                    if round_file.name.endswith('.csv'):
                        round_df = pd.read_csv(round_file)
                    elif round_file.name.endswith('.xlsx'):
                        round_df = pd.read_excel(round_file)
                except Exception as e:
                    st.error(f"Error loading round gemstone data: {str(e)}")
            
            if nonround_file:
                try:
                    if nonround_file.name.endswith('.csv'):
                        nonround_df = pd.read_csv(nonround_file)
                    elif nonround_file.name.endswith('.xlsx'):
                        nonround_df = pd.read_excel(nonround_file)
                except Exception as e:
                    st.error(f"Error loading non-round gemstone data: {str(e)}")
        
        price_calculator = GemstonePriceCalculator(round_df, nonround_df)
        
        # Step 1: Get product data from Channel Advisor
        step1_status.info("🔄 Step 1: Fetching product data from Channel Advisor...")
        raw_product_data = ca_api.get_product_by_sku(sku_input)
        
        if raw_product_data:
            step1_status.success("✅ Step 1: Product data retrieved successfully")
            
            # Step 2: Extract ALL attributes and gemstone data
            step2_status.info("🔄 Step 2: Extracting all attributes and processing gemstone data...")
            final_attributes, gemstone_details, structured_info, comprehensive_summary = process_with_complete_extraction(raw_product_data, GEMINI_API_KEY)
            step2_status.success("✅ Step 2: All attributes extracted and gemstone data processed")
            
            # Step 3: Calculate gemstone costs
            step3_status.info("🔄 Step 3: Calculating gemstone costs...")
            bill_data = price_calculator.calculate_gemstone_bill(gemstone_details)
            step3_status.success("✅ Step 3: Gemstone cost calculation completed")
            
            # Display results
            st.subheader("💎 Complete Product Analysis & Cost Calculation")

            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["All Attributes", "Gemstone Analysis", "Bill Calculation", "Raw Data"])

            with tab1:
                st.write("**Complete Channel Advisor Attributes**")
                
                # Display all extracted attributes in organized sections
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"- ID: {final_attributes.id}")
                    st.write(f"- Title: {final_attributes.title}")
                    st.write(f"- Description: {final_attributes.description}")
                    
                    st.write("**Pick Fields:**")
                    st.write(f"- Pick 1: {final_attributes.pick_1}")
                    st.write(f"- Pick 2: {final_attributes.pick_2}")
                    st.write(f"- Pick 3: {final_attributes.pick_3}")
                    
                    st.write("**Material & Metal:**")
                    st.write(f"- Material: {final_attributes.material}")
                    st.write(f"- Metal: {final_attributes.metal}")
                    st.write(f"- Metal Purity: {final_attributes.metal_purity}")
                
                with col2:
                    st.write("**Measurements:**")
                    st.write(f"- Weight: {final_attributes.weight}")
                    st.write(f"- Width: {final_attributes.width}")
                    st.write(f"- Size: {final_attributes.size}")
                    st.write(f"- Casting Weight: {final_attributes.casting_weight}")
                    st.write(f"- Right Weight: {final_attributes.right_weight}")
                    
                    st.write("**Stone Information:**")
                    st.write(f"- Stone Type: {final_attributes.stone_type}")
                    st.write(f"- Stone Shape: {final_attributes.stone_shape}")
                    st.write(f"- Stone Dimensions: {final_attributes.stone_dimensions}")
                    st.write(f"- Stone Carat: {final_attributes.stone_carat}")
                    st.write(f"- Stone Quantity: {final_attributes.stone_quantity}")
                    
                    st.write("**Diamond Information:**")
                    st.write(f"- Number of Diamonds: {final_attributes.number_of_diamonds}")
                    st.write(f"- Diamond Carat: {final_attributes.diamond_carat}")
                    
                    st.write("**Other:**")
                    st.write(f"- Main Stone: {final_attributes.main_stone}")
                    st.write(f"- Gender: {final_attributes.gender}")
                    st.write(f"- Exact Carat Total Weight: {final_attributes.exact_carat_total_weight}")

            with tab2:
                st.write("**Product Summary**")
                
                # Display the comprehensive summary
                st.info(comprehensive_summary)
                
                st.write("---")
                
                st.write("**Extracted Gemstone Details**")
                
                if gemstone_details:
                    for i, gem in enumerate(gemstone_details, 1):
                        st.write(f"**Gemstone {i}:**")
                        
                        # Create columns for better display
                        gem_col1, gem_col2 = st.columns(2)
                        
                        with gem_col1:
                            st.write(f"- **Type:** {gem.gemstone_type}")
                            st.write(f"- **Shape:** {gem.gemstone_shape}")
                            st.write(f"- **Size:** {gem.gemstone_size}")
                        
                        with gem_col2:
                            st.write(f"- **Quantity:** {gem.quantity}")
                            st.write(f"- **Carat:** {gem.gemstone_carat}")
                            if gem.product_id:
                                st.write(f"- **Product ID:** {gem.product_id}")
                        
                        st.write("---")
                
                else:
                    st.warning("No gemstone details extracted.")
                    st.write("**Possible reasons:**")
                    st.write("- Product contains only diamonds (excluded from gemstone analysis)")
                    st.write("- Stone Type attribute is empty or marked as 'N/A'")
                    st.write("- No recognizable gemstone information found in product data")
                
                # Show extraction method used
                st.write("**Extraction Analysis:**")
                if final_attributes.stone_type and final_attributes.stone_type.lower() not in ['diamond', 'n/a', 'not available', '']:
                    st.success("✅ Direct extraction from Channel Advisor attributes")
                elif final_attributes.pick_2:
                    st.info("ℹ️ Extracted from Pick fields analysis")
                else:
                    st.warning("⚠️ Used Gemini AI fallback extraction")

            with tab3:
                st.write("**Gemstone Bill Calculation**")
                
                if bill_data.get('gemstones'):
                    # Create detailed bill table
                    bill_df_data = []
                    for gem in bill_data['gemstones']:
                        bill_df_data.append({
                            'Gemstone': f"{gem['gemstoneType']} ({gem['gemstoneShape']})",
                            'Size': gem['gemstoneSize'],
                            'Quantity': gem['quantity'],
                            'Carat': f"{gem['gemstoneCarat']:.2f}",
                            'Unit Price': f"${gem['unit_price']:.2f}",
                            'Subtotal': f"${gem['total_bill']:.2f}",
                            'Product ID': gem.get('productId', 'N/A'),
                            'Note': gem.get('note', '')
                        })
                    
                    bill_df = pd.DataFrame(bill_df_data)
                    st.dataframe(bill_df, use_container_width=True)
                    
                    # Display total
                    st.markdown(f"### **Total Gemstone Bill: ${bill_data['total_bill']:.2f}**")
                    
                    # Display summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Gemstones", len(bill_data['gemstones']))
                    with col2:
                        total_quantity = sum(gem['quantity'] for gem in bill_data['gemstones'])
                        st.metric("Total Quantity", total_quantity)
                    with col3:
                        avg_price = bill_data['total_bill']/total_quantity if total_quantity > 0 else 0
                        st.metric("Average Price/Stone", f"${avg_price:.2f}")
                    
                    # Additional breakdown
                    st.write("**Cost Breakdown:**")
                    for i, gem in enumerate(bill_data['gemstones']):
                        percentage = (gem['total_bill'] / bill_data['total_bill'] * 100) if bill_data['total_bill'] > 0 else 0
                        st.write(f"- {gem['gemstoneType']} ({gem['gemstoneShape']}): ${gem['total_bill']:.2f} ({percentage:.1f}%)")
                
                else:
                    st.warning("No gemstone bill data available")
                    if bill_data.get('error'):
                        st.error(f"Error: {bill_data['error']}")
                    
                    # Show what was processed for debugging
                    if gemstone_details:
                        st.info("Gemstone details were found but no pricing data was available:")
                        for i, gem in enumerate(gemstone_details):
                            st.write(f"- Gemstone {i+1}: {gem.gemstone_type} ({gem.gemstone_shape}) - {gem.gemstone_size}")
                    else:
                        st.error("No gemstone details were extracted from the product data.")

            with tab4:
                st.write("**Raw Channel Advisor API Response**")
                st.json(raw_product_data)
                
                st.write("**Structured Processing Summary**")
                st.text(structured_info)
        
        else:
            step1_status.error("❌ Step 1: Failed to retrieve product data")
            st.error(f"Could not find product with SKU: {sku_input}")
            st.info("Please check:")
            st.info("- SKU is correct and exists in Channel Advisor")
            st.info("- API credentials are valid")
            st.info("- Network connection is working")

if __name__ == "__main__":
    main()
