from mac_transcriber.chatgpt_auth import (
    derive_chatgpt_bridge_api_key,
    load_control_plane_api_key,
    load_chatgpt_bridge_api_key,
)


def test_bridge_key_derivation_is_stable_and_not_the_control_plane_key():
    control_plane_key = "test-control-plane-key"

    first = derive_chatgpt_bridge_api_key(control_plane_key)
    second = derive_chatgpt_bridge_api_key(control_plane_key)

    assert first == second
    assert len(first) == 64
    assert first != control_plane_key
    assert derive_chatgpt_bridge_api_key("") == ""


def test_loader_reads_only_the_named_control_plane_value(tmp_path):
    env_file = tmp_path / "tunnel.env"
    env_file.write_text(
        "MAC_TRANSCRIBER_API_KEY=operator-key\n"
        "UNRELATED_SECRET=do-not-use\n"
        "CONTROL_PLANE_API_KEY=bridge-root\n",
        encoding="utf-8",
    )

    assert load_chatgpt_bridge_api_key(env_file) == derive_chatgpt_bridge_api_key(
        "bridge-root"
    )
    assert load_control_plane_api_key(env_file) == "bridge-root"
