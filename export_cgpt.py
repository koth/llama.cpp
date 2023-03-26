import torch
import argparse
import struct
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description='Convert a LLaMA model checkpoint to a ggml compatible file')
    parser.add_argument('model_path',  help='the model path of checkpoint')
    parser.add_argument('output_path',  help='the output path of model')
    parser.add_argument('ftype',      help='file type (0: float32, 1: float16)', type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main():

    args = parse_args()
    model_path = args.model_path
    ftype = args.ftype
    ftype_str = ["f32", "f16"]

    print(args)
    fout = open(args.output_path,"wb")
    model = torch.load(model_path, map_location="cpu")


    def write_header(shape, dst_name, ftype_cur):
        sname = dst_name.encode('utf-8')
        fout.write(struct.pack("iii", len(shape), len(sname), ftype_cur))
        fout.write(struct.pack("i" * len(shape), *shape[::-1]))
        fout.write(sname)

    def convert(src_name, dst_name,permute=False):
        v = model[src_name]
        shape = v.shape
        print("Processing variable: " + src_name + " with shape: ", shape, " and type: ", v.dtype)
        if len(shape) == 1:
            print("  Converting to float32")
            v = v.to(torch.float32)

        ftype_cur = {torch.float16: 1, torch.float32: 0}[v.dtype]

        # header
        write_header(shape, dst_name, ftype_cur)
        if permute:
            v = v.contiguous().view(32,2,2048//32//2,2048).transpose(1,2).reshape(2048,2048)


        # data
        v.numpy().tofile(fout)

    fout.write(struct.pack("i", 0x67676d66)) # magic: ggmf in hex
    fout.write(struct.pack("i", 1)) # file version
    fout.write(struct.pack("i", 51200)) # n vocab
    fout.write(struct.pack("i", 2048)) # hidden size
    fout.write(struct.pack("i", 1)) #multiple of part
    fout.write(struct.pack("i", 32)) # num of head
    fout.write(struct.pack("i", 24)) # num of layer
    fout.write(struct.pack("i", 2048 // 32)) # rot (obsolete)
    fout.write(struct.pack("i", ftype))
    convert("model.embed_tokens.weight", "tok_embeddings.weight")
    convert("model.norm.weight", "norm.weight")
    convert("lm_head.weight", "output.weight")

    for i in range(24):
        convert(f"model.layers.{i}.self_attn.q_proj.weight", f"layers.{i}.attention.wq.weight",permute=True)
        convert(f"model.layers.{i}.self_attn.k_proj.weight", f"layers.{i}.attention.wk.weight",permute=True)
        convert(f"model.layers.{i}.self_attn.v_proj.weight", f"layers.{i}.attention.wv.weight")
        convert(f"model.layers.{i}.self_attn.o_proj.weight", f"layers.{i}.attention.wo.weight")

        convert(f"model.layers.{i}.mlp.gate_proj.weight", f"layers.{i}.feed_forward.w1.weight")
        convert(f"model.layers.{i}.mlp.down_proj.weight", f"layers.{i}.feed_forward.w2.weight")
        convert(f"model.layers.{i}.mlp.up_proj.weight",   f"layers.{i}.feed_forward.w3.weight")

        convert(f"model.layers.{i}.input_layernorm.weight", f"layers.{i}.attention_norm.weight")
        convert(f"model.layers.{i}.post_attention_layernorm.weight", f"layers.{i}.ffn_norm.weight")


    fout.close()

    print("Done. Output file: " + args.output_path)


if __name__ == "__main__":
    main()
