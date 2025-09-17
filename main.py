from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
import os
import re
from typing import List, Optional
import razorpay
from dotenv import load_dotenv
import requests
import aiohttp
import asyncio

# --- Load Environment Variables ---
load_dotenv()
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")


# --- Razorpay Client ---
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# --- Data Models (Add new ones) ---
class PaymentRequest(BaseModel):
    amount: int
    receipt_id: str

class OrderResponse(BaseModel):
    order_id: str
    amount: int
    currency: str

class ChatMessage(BaseModel):
    role: str
    text: str

class ConversationRequest(BaseModel):
    history: List[ChatMessage]
    prompt: str

class TripSuggestion(BaseModel):
    location: Optional[str] = None
    description: Optional[str] = None
    budget: Optional[str] = None
    error: Optional[str] = None

class Recommendation(BaseModel):
    placeName: str
    imageUrl: Optional[str] = None
    reason: Optional[str] = None 
    
async def fetch_image(session, place_name: str) -> Optional[Recommendation]:
    """
    Searches Pexels for a photo of the given place and returns an image URL.
    """
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": place_name, "per_page": 1}
    try:
        async with session.get(url, headers=headers, params=params, timeout = 10) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("photos"):
                    return Recommendation(placeName=place_name, imageUrl=data["photos"][0]["src"]["medium"])
    except Exception as e:
        print(f"Could not fetch image for {place_name}. Error: {e}")
    
    return None

# --- FastAPI App & Gemini Config (Unchanged) ---
app = FastAPI()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set!")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# --- Add new endpoint for payment orders ---
@app.post("/create-razorpay-order", response_model=OrderResponse)
def create_order(request: PaymentRequest):
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay keys not configured.")
    
    order_data = {
        "amount": request.amount, # Amount in paise (e.g., 500 for Rs. 5.00)
        "currency": "INR",
        "receipt": request.receipt_id,
        "payment_capture": 1
    }
    
    try:
        order = razorpay_client.order.create(data=order_data)
        return OrderResponse(
            order_id=order["id"],
            amount=order["amount"],
            currency=order["currency"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- The main AI chat endpoint ---
@app.post("/generate-suggestion")
def generate_suggestion(request: ConversationRequest):
    """
    Handles a full conversational turn with the user, letting the AI
    ask for any missing information before providing a suggestion.
    """
    
    current_history = request.history + [ChatMessage(role="user", text=request.prompt)]
    
    gemini_history_text = "\n".join([f"{msg.role}: {msg.text}" for msg in current_history])
    full_prompt = (
        "You are an expert, conversational, budget-friendly travel guide. "
        "Your goal is to suggest ONE single, specific travel destination to the user. "
        "To do this, you MUST first know where the user is traveling from and the trip duration. "
        "If you have both, your response MUST be structured as below. You MUST include all three sections. "
        "DO NOT use bold formatting.\n\n"
        "Location: [destination]\n"
        "Description: [description]\n"
        "Budget: [budget]\n\n"
        f"Chat History & Latest Request:\n{gemini_history_text}"
    )

    gemini_history = [{"role": msg.role, "parts": [msg.text]} for msg in current_history]

    try:
        chat = model.start_chat(history=gemini_history[:-1])
        response = chat.send_message(full_prompt)
        text = response.text
        
        if "Location:" in text and "Description:" in text and "Budget:" in text:
            location_match = re.search(r"Location: (.+)", text)
            
            description_match = re.search(r"Description: (.+?)(?=Budget:)", text, re.DOTALL)
            
            budget_match = re.search(r"Budget: (.+)", text, re.DOTALL)
            
            return TripSuggestion(
                location=location_match.group(1).strip() if location_match else None,
                description=description_match.group(1).strip() if description_match else None,
                budget=budget_match.group(1).strip() if budget_match else None
            )
        else:
            return TripSuggestion(description=text)

    except Exception as e:
        print(f"An error occurred: {e}")
        return TripSuggestion(error="Sorry, an error occurred with the AI.")
    
@app.get("/recommendations", response_model=List[Recommendation])    
async def get_recommendations():
    """
    Returns a list of travel recommendations with dynamic, real-world images.
    """
    places_to_recommend = [
        "Varanasi, India",
        "Paris, France",
        "Kyoto, Japan",
        "Rome, Italy",
        "Cairo, Egypt",
        "Prague, Czechia"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, place) for place in places_to_recommend]
        results = await asyncio.gather(*tasks)
        
    return results

# --- Async function to get trending place from AI ---
async def get_trending_place() -> str:
    try:
        prompt = (
            "You are a travel assistant. Suggest ONE popular tourist destination"
            "that has been trending recently based on maximum tourist visits."
            "The place should be a well-known location"
            "take current time and season into account too."
            "take into account any upcoming or ongoing local festivals or events."
            "take into account the current weather conditions."
            "take into account every seconds and minutes."
            "that is best to visit right now, based on weather, events, and local festivals."
            "that is not a city or state. Format: Place, Country."
            "Also give a short explanation (1-2 sentences) why it is trending. "
            "Return the place first, then the reason separated by a | character."
            "Do not include symbols like ** or markdown formatting."
            "give me maximum 200 words of reason."
            "just give me place name and reason. nothing else"
        )

        response = model.generate_content(prompt)
        text = response.text.strip()

        if "|" in text:
            place, reason = map(str.strip, text.split("|", 1))
            return place, reason

        return text, "Trending destination"
    except Exception as e:
        print(f"Error fetching trending place: {e}")
        return "Prayagraj, India"  # fallback

# --- Endpoint: Trending Recommendation ---
@app.get("/trending-recommendation", response_model=Recommendation)
async def trending_recommendation():
    place,reason = await get_trending_place()
    async with aiohttp.ClientSession() as session:
        rec = await fetch_image(session, place)
    rec.reason = reason
    return rec