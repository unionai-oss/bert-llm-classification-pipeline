from flytekit import ImageSpec, Resources
from union.actor import ActorEnvironment

container_image = ImageSpec(
    requirements="requirements.txt",
)

actor = ActorEnvironment(
    name="my-actor",
    container_image=container_image,
    replica_count=1,
    ttl_seconds=360,
    requests=Resources(
        cpu="2",
        mem="5000Mi",
        gpu="1",
    ),
)