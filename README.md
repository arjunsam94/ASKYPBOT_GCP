# ASKYP RAG CHATBOT

## Problem
### Time 
The YPCommunities team produces over 30 market research documents identifying new community engagement and partnership opportunities. This forced the members to sift through multiple documents to get helpful insights and develop strategies.
### Cost
The seat-based pricing model makes it prohibitively expensive for a non-profit to use Gemini with Google workspaces since they rely on adding new volunteers and regional managers to help with projects on an ongoing basis.

No additional budget for new tools, cloud services, or multiple engineering resources to build and maintain the system.

## Proof of Concept
Built a POC on the local machine to get buy-in from the non-profit board members.

![POC](https://github.com/arjunsam94/ASKYPBOT_GCP/blob/ac37707815bf3b3ac3eaaa3c0ed746c54b223fbb/POC_updated.png)

## Cloud solution
Deployed on the cloud with Docker, Cloud run, Cloud Build, and Firestore to open the availability for the rest of the team.

![Cloud](https://github.com/arjunsam94/ASKYPBOT_GCP/blob/23daa7a2fa5fba5cfae2f706506c9aba0bdf20bd/Cloud.png)

## Demo

[![Demo](https://github.com/arjunsam94/ASKYPBOT_GCP/blob/e5b55ab8536e1ca40f3d8478302e7bcf486494a2/ASKYPdemo.png)](https://youtu.be/2wU9EjaIsPo)

## Thought process

- Chose to run the models locally using Ollama with Gemma3, Nomic text embed, and ChromaDB over using APIs during the POC stage. This enabled testing without worrying about exceeding the free API limits even though the output was slower.
- Chose to build the solution instead of getting access to Gemini features within the Google workspace since it required an upgrade to the paid plan that doesn’t offer non-profit discount.
- Decided to use Gemini Flash 1.5 and text embedding API over using Ollama on Runpod, since the reports were compiled with mostly publicly available information and added complexity for more security was not necessary.
- Decided not to go with multimodal embedding models even though we had both text and images in our documents since the generated insights were primarily based on text data, as observed during testing.
- Split data ingestion and response generation into separate Docker containers instead of combining within one to parallelize tasks for improved speed and to avoid application failure when no new files are available for ingestion.

## Impact

- Reduced the time taken to compile insights about sourcing partners and engagement strategies from hours to a few minutes.
- Saved over $1000 in total subscription cost without requiring additional engineering resources.
- Reduced data latency by 4 seconds per request (over 65%) by building and deploying separate Docker containers to handle data ingestion and response generation tasks in parallel.
