import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent
from relari_otel import telemetry_init
from relari_otel.runners import async_eval_runner, batch_eval_runner
from relari_otel.dataset import Dataset, Scenario, Contract, NLRequirement, Requirements, DeterministicRequirement
import os

# Initialize telemetry
telemetry_init(
	project_name="browser-use-demo",
	# exclude_instrumentators=["openai"],
	batch=False,
	api_key="ek-a53d7330a74ecfc35310fd2497f294bd"
)


load_dotenv()

# Initialize the model
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)
jaeger_task = "Go to http://localhost:16686/ and find the trace with the service relari-otel around 10:24am today. Then find the spans named ChatOpenAI and extract the prompt and response."
medusa_task = "Go to http://localhost:9000/app, wait for user login credentials, and then create a new product with random details."
maps_task = "Go to Google Maps, search for restaurants near Caltrain station at 4th and King St in San Francisco, and find one that's open for brunch on Friday within walking distance."

# Create contract for the browser automation tasks
jaeger_contract = Contract(
    name="jaeger_trace_extraction",
    condition=NLRequirement(
        requirement="Access Jaeger UI and extract ChatOpenAI trace information"
    ),
    preconditions=Requirements(
        must=[
            NLRequirement(
                requirement="Verify Jaeger UI is accessible at http://localhost:16686/"
            ),
            NLRequirement(
                requirement="Jaeger UI service must be running and accessible"
            )
        ]
    ),
    postconditions=Requirements(
        must=[
            NLRequirement(
                requirement="Find trace with service 'relari-otel' and span name 'ChatOpenAI'"
            ),
            NLRequirement(
                requirement="Successfully extracted prompt and response from ChatOpenAI spans"
            )
        ]
    ),
    constraints=Requirements(
        must=[
            NLRequirement(
                requirement="Search for traces around 10:24am on the current day"
            )
        ]
    )
)

# Create contract for the Medusa task
medusa_contract = Contract(
    name="medusa_product_creation",
    condition=NLRequirement(
        requirement="Access Medusa admin panel and create a new product"
    ),
    preconditions=Requirements(
        must=[
            NLRequirement(
                requirement="Verify Medusa admin panel is accessible at http://localhost:9000/app"
            ),
            NLRequirement(
                requirement="User must provide login credentials"
            )
        ]
    ),
    postconditions=Requirements(
        must=[
            NLRequirement(
                requirement="Successfully logged into Medusa admin panel"
            ),
            NLRequirement(
                requirement="Created a new product with random details"
            )
        ]
    ),
    constraints=Requirements(
        must=[
            NLRequirement(
                requirement="Must pause and wait for user input before proceeding with login"
            )
        ]
    )
)

# Create contract for the Google Maps restaurant search
maps_contract = Contract(
    name="restaurant_search",
    condition=NLRequirement(
        requirement="Find a suitable restaurant near Caltrain that meets all criteria"
    ),
    preconditions=Requirements(
        must=[
            NLRequirement(
                requirement="Verify Google Maps is accessible at https://www.google.com/maps"
            ),
            NLRequirement(
                requirement="Google Maps must be accessible and functional"
            )
        ]
    ),
    postconditions=Requirements(
        must=[
            NLRequirement(
                requirement="Verify search is centered on 4th and King St Caltrain Station"
            ),
            NLRequirement(
                requirement="Found restaurant must be open for brunch on Fridays"
            ),
            NLRequirement(
                requirement="Restaurant must be within 15 minutes walking distance from Caltrain"
            ),
            NLRequirement(
                requirement="Verify restaurant's operating hours include Friday brunch time (typically 10am-2pm)"
            )
        ]
    ),
    constraints=Requirements(
        must=[
            NLRequirement(
                requirement="Must check actual walking route distance, not just straight-line distance"
            ),
            NLRequirement(
                requirement="Must verify current operating hours, not relying on outdated information"
            )
        ]
    )
)

# Create scenarios and dataset for telemetry
jaeger_scenario = Scenario(
    uuid="jaeger-task", 
    data={"task": jaeger_task},
    contracts=[jaeger_contract]
)

medusa_scenario = Scenario(
    uuid="medusa-task",
    data={"task": medusa_task},
    contracts=[medusa_contract]
)

# Create scenario for maps task
maps_scenario = Scenario(
    uuid="maps-restaurant-search",
    data={"task": maps_task},
    contracts=[maps_contract]
)

# Update dataset to include all scenarios
dataset = Dataset(scenarios=[
    # jaeger_scenario,
    # medusa_scenario,
    maps_scenario
])

async def run_agent(data, context=None):
	"""Wrapper function for agent.run() to use with async_eval_runner"""
	# Create a new agent for each scenario using the task from the data
	agent = Agent(task=data["task"], llm=llm)
	await agent.run()

async def main():
	# Use batch_eval_runner instead of async_eval_runner
	await async_eval_runner(
		dataset,
		run_agent,
	)
	# await batch_eval_runner(
	# 	dataset, 
	# 	run_agent, 
	# 	context_factory=lambda _: {}, 
	# 	batch_size=3
	# )

if __name__ == '__main__':
	asyncio.run(main())
