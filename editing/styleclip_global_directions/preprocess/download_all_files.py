from pathlib import Path
import subprocess
import os

PATHS = {
    "sg3-r-ffhq-1024": {"delta_i_c.npy": "1HOUGvtumLFwjbwOZrTbIloAwBBzs2NBN",
                        "s_stats": "1FVm_Eh7qmlykpnSBN1Iy533e_A2xM78z"},
    "sg3-r-ffhqu-1024": {"delta_i_c.npy": "1EcLy3ya7p-cWs8kQZKgudyZnhGvsrRUO",
                         "s_stats": "1It-M23K31ABgGiH7CAUfmmj_SEqFMfM_"},
    "sg3-r-afhq-512": {"delta_i_c.npy": "1CKDn0BcbAosGLEYo4fW2YnAyaERvtJ7s",
                       "s_stats": "1omJCjPSyamP01Pr1rPx0wO4eI1Jpohat"},
    "sg3-t-landscape-256": {"delta_i_c.npy": "1Po4S_zPuefQZFttT4tW9z7dt-nu4P4iF",
                            "s_stats": "12XqJ4DX31n2AtVpPFZXfOiUxTUJ5CxhK"}
}


def main():
    save_dir = Path("editing") / "styleclip_global_directions"
    save_dir.mkdir(exist_ok=True, parents=True)
    for name, file_ids in PATHS.items():
        model_dir = save_dir / name
        model_dir.mkdir(exist_ok=True, parents=True)
        print(f"Downloading models for {name}...")
        for file_name, file_id in file_ids.items():
            subprocess.run(["gdown", "--id", file_id, "-O", model_dir / file_name])
        # remove extra files
        try:
            for path in model_dir.glob("*"):
                if str(path.name).startswith("sg3"):
                    os.remove(path)
        except Exception:
            pass


if __name__ == '__main__':
    main()
