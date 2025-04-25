from flytekit import ImageSpec, Resources
from union.actor import ActorEnvironment

container_image = ImageSpec(
     name="fine-tune-qlora",
    requirements="requirements.txt",
    pip_extra_index_url=["https://download.pytorch.org/whl/cu118"],  # ✅ enables +cu118 builds
    builder="union",
    cuda="11.8",  # ensure GPU + CUDA layer is available
    apt_packages=["gcc", "g++"],  # optional, for packages like bitsandbytes
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
