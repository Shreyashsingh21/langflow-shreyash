from langflow.custom import Component
from langflow.io import (
    StrInput, IntInput, TabInput,
    Output
)
from langflow.schema.message import Message
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Type

import base64
import json
import time
import re
import requests


class TicketContextInput(BaseModel):
    """Input schema for the ticket context tool."""
    action: str = Field(description="The action name")
    action_input: dict = Field(description="The action input containing ticket_id")
    
    def get_ticket_id(self) -> str:
        """Extract ticket_id from action_input."""
        return str(self.action_input.get('ticket_id', ''))


class TrinkaSupportTicketContext(Component):
    display_name = "Trinka Ticket Context Tool"
    description = "Fetches Freshdesk ticket context and outputs structured JSON for agents to use."
    icon = "Ticket"
    category = "tools"

    inputs = [
        StrInput(name="ticket_id", display_name="Ticket ID", required=False),
        IntInput(name="ticket_count", display_name="Ticket Count", value=5),
        TabInput(name="mode", display_name="Mode", options=["Parser", "Stringify"], value="Stringify"),
    ]

    outputs = [
        Output(name="ticket_context", display_name="Ticket Context", method="build"),
        Output(name="tool", display_name="Tool", method="build_tool", types=["Tool"]),
    ]

    FRESHDESK_DOMAIN = "trinka.freshdesk.com"
    API_KEY = "Vs3cRt89YV2MX6sT7Dvw"

    def _auth_headers(self):
        auth = f"{self.API_KEY}:X"
        encoded = base64.b64encode(auth.encode()).decode()
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json"
        }

    def _get_requester_id(self, ticket_id: str) -> str:
        url = f"https://{self.FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
        resp = requests.get(url, headers=self._auth_headers())
        resp.raise_for_status()
        return str(resp.json().get("requester_id", ""))

    def _get_tickets_with_conversations(self, requester_id: str, count: int):
        url = f"https://{self.FRESHDESK_DOMAIN}/api/v2/tickets?requester_id={requester_id}&include=description&order_by=created_at&order_type=desc&per_page=100"
        resp = requests.get(url, headers=self._auth_headers())
        resp.raise_for_status()
        all_tickets = resp.json()
        selected_tickets = all_tickets[:count]

        # Attach conversations
        for ticket in selected_tickets:
            try:
                t_id = ticket["id"]
                conv_url = f"https://{self.FRESHDESK_DOMAIN}/api/v2/tickets/{t_id}/conversations"
                conv_resp = requests.get(conv_url, headers=self._auth_headers())
                conv_resp.raise_for_status()
                ticket["conversations"] = conv_resp.json()
            except Exception as e:
                ticket["conversations"] = []
                ticket["conversation_error"] = str(e)
            time.sleep(0.5)

        return {"requester_id": requester_id, "tickets": selected_tickets}

    def _sanitize_subject(self, subject: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', '', subject).strip()

    def _generate_context(self, tickets_json: dict):
        try:
            tickets = tickets_json.get("tickets", [])
            requester_id = tickets_json.get("requester_id", "unknown")

            conversations_context = []
            for t in tickets:
                ticket_id = t.get("id")
                subject = self._sanitize_subject(t.get("subject", ""))
                conversations = []

                if "conversations" in t and t["conversations"]:
                    for conv in t["conversations"]:
                        conversations.append({
                            "ticket_id": ticket_id,
                            "requester_id": requester_id,
                            "description_text": conv.get("body_text") or conv.get("description") or ""
                        })
                else:
                    # Fallback to description_text if no conversations
                    conversations.append({
                        "ticket_id": ticket_id,
                        "requester_id": requester_id,
                        "description_text": t.get("description_text", "")
                    })

                conversations_context.append({
                    "ticket_id": ticket_id,
                    "subject": subject,
                    "conversations": conversations
                })

            return {
                "requester_id": requester_id,
                "total_tickets": len(tickets),
                "conversations_context": conversations_context
            }
        except Exception as e:
            return {"error": str(e)}

    def _fetch_ticket_context(self, ticket_id: str) -> str:
        """Internal method to fetch ticket context - used by both build methods."""
        try:
            if not ticket_id:
                return json.dumps({"error": "No ticket ID provided"})
            
            requester_id = self._get_requester_id(ticket_id)
            tickets_json = self._get_tickets_with_conversations(requester_id, self.ticket_count)
            context_data = self._generate_context(tickets_json)
            return json.dumps(context_data, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Exception: {str(e)}"})

    def build(self) -> Message:
        """Build method for direct component usage."""
        if not self.ticket_id:
            return Message(text=json.dumps({"error": "No ticket ID provided"}))
        
        result = self._fetch_ticket_context(self.ticket_id)
        return Message(text=result)

    def build_tool(self) -> StructuredTool:
        """Build method that returns a StructuredTool for agent usage."""
        
        def fetch_context(action: str, action_input: dict) -> str:
            """Fetch Freshdesk ticket context including related tickets and conversations.
            
            Args:
                action: The action name
                action_input: The action input containing ticket_id
                
            Returns:
                JSON string containing ticket context data
            """
            ticket_id = str(action_input.get('ticket_id', ''))
            if not ticket_id:
                return json.dumps({"error": "No ticket ID provided"})
                
            return self._fetch_ticket_context(ticket_id)

        return StructuredTool.from_function(
            func=fetch_context,
            name="trinka_ticket_context",
            description="Fetch Freshdesk ticket context as structured JSON. Use this tool ONCE per ticket ID to get complete information about a specific ticket and related tickets from the same requester. Do not call this tool repeatedly for the same ticket ID. Provide the ticket ID as input.",
            args_schema=TicketContextInput,
        )