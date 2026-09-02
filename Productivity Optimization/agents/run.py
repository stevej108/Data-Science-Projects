from agents.workflow import app
from core.state import init_state


def run_workflow(
    query,
    onedrive_root,
    top_k,
    memo_output_path="./output",
):

    state = init_state(
        query=query,
        onedrive_root=onedrive_root,
        top_k=top_k,
        memo_output_path=memo_output_path,
    )

    final_state = {}

    progress_count = 0

    for update in app.stream(
        state,
        stream_mode="updates",
    ):

        #
        # Merge node outputs
        #
        for value in update.values():

            if isinstance(value,dict):
                final_state.update(value)

        #
        # Emit new progress messages
        #
        progress = final_state.get("progress", [])

        while progress_count < len(progress):

            yield {
                "__progress__": progress[progress_count]
            }

            progress_count += 1

    #
    # ONLY send final state once
    #
    yield {
        "__final__": final_state
    }