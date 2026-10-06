import asyncio
import httpx

async def test_api():
    async with httpx.AsyncClient() as client:
        print("1. Testing Login...")
        response = await client.post(
            "http://127.0.0.1:8000/api/auth/token",
            data={"username": "arjun@chronofi.demo", "password": "password123"}
        )
        if response.status_code != 200:
            print("Login Failed:", response.text)
            return
            
        token = response.json()["access_token"]
        print("Successfully obtained JWT token!")
        
        headers = {"Authorization": f"Bearer {token}"}
        
        print("\n2. Testing Protected Profile Route...")
        profile_response = await client.get("http://127.0.0.1:8000/api/profile/me", headers=headers)
        print("Profile Data:", profile_response.json())
        
        print("\n3. Testing SSE Chat Streaming Route...")
        async with client.stream(
            "POST", 
            "http://127.0.0.1:8000/api/chat", 
            headers=headers,
            json={"user_id": 1, "message": "What is my current savings and how does it look for my car goal?"}
        ) as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    print(line)

if __name__ == "__main__":
    asyncio.run(test_api())
