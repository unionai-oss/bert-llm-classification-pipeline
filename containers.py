from flytekit import ImageSpec, Resources
from union.actor import ActorEnvironment

container_image = ImageSpec(
    requirements="requirements.txt",
    builder="union",
    apt_packages=["gcc", "g++"],
    cuda="11.8",
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
