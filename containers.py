from flytekit import ImageSpec, Resources
from union.actor import ActorEnvironment

container_image = ImageSpec(
    requirements="requirements.txt",
)

# actor = ActorEnvironment(
#     name="my-actor",
#     container_image=image,
#     replica_count=1,
#     ttl_seconds=120,
#     requests=Resources(
#         cpu="2",
#         mem="500Mi",
#     ),
# )