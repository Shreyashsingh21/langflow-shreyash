from langflow.custom import Component
from langflow.io import (
    StrInput, IntInput, TabInput,
    Output
)
from langflow.schema.message import Message
from langchain.tools import Tool

import base64
import json
import time
import re
import requests


class TrinkaSupportTicketContext(Component):
    display_name = "Trinka Ticket Context Tool"
    description = "Fetches Freshdesk ticket context and outputs structured JSON for agents to use."
    icon = "📝"

    inputs = [
        StrInput(name="ticket_id", display_name="Ticket ID", required=True),
        IntInput(name="ticket_count", display_name="Ticket Count", value=5),
        TabInput(name="mode", display_name="Mode", options=["Parser", "Stringify"], value="Stringify"),
    ]

    outputs = [
        Output(name="ticket_context", display_name="Ticket Context", method="build"),
        Output(name="tool", display_name="Tool", method="build_tool"),
    ]

    FRESHDESK_DOMAIN = "trinka.freshdesk.com"
    API_KEY = "Vs3cRt89YV2MX6sT7Dvw"
    pattern = "Text: {text}"

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

    def build(self) -> Message:
        try:
            ticket_id = self.ticket_id
            if not ticket_id:
                return Message(text="No ticket ID provided.")

            requester_id = self._get_requester_id(ticket_id)
            tickets_json = self._get_tickets_with_conversations(requester_id, self.ticket_count)
            context_data = self._generate_context(tickets_json)

            return Message(text=json.dumps(context_data, indent=2))
        except Exception as e:
            return Message(text=f"Unhandled Exception: {str(e)}")

    def build_tool(self) -> Tool:
        def tool_func(input_data) -> str:
            try:
                # Handle both string input and structured input
                if isinstance(input_data, str):
                    ticket_id = input_data
                elif isinstance(input_data, dict):
                    ticket_id = input_data.get('ticket_id', '')
                else:
                    ticket_id = str(input_data)
                
                if not ticket_id:
                    return "Error: No ticket ID provided"
                
                requester_id = self._get_requester_id(ticket_id)
                tickets_json = self._get_tickets_with_conversations(requester_id, 5)
                context_data = self._generate_context(tickets_json)
                return json.dumps(context_data, indent=2)
            except Exception as e:
                return f"Tool Error: {str(e)}"

        return Tool(
            name="Trinka_ticket_context_tool",
            func=tool_func,
            description="Fetch Freshdesk ticket context as structured JSON. Input should be a ticket ID."
        )
