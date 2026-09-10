"""Two independent opt-in Q8 R8 changes; preserve all other kernel paths."""


def transform(source):
    marker='static int q8_0_vnni_x_tile() {'
    assert source.count(marker)==1
    start=source.index(marker);end=source.index('\nvoid ggml_gemv_q8_0_8x8_q8_0(',start)
    body=source[start:end]
    assert body.count('    return value;')==1
    body=body.replace('    return value;', '''    static const bool ssm_tile8=[] {
        const char * text=std::getenv("GGML_CPU_Q8_R8_SSM_TILE8");
        return text && std::strcmp(text,"1")==0;
    }();
    return n==1536 && ssm_tile8 ? 8 : value;''')
    source=source[:start]+body+source[end:]
    assert source.count('q8_0_vnni_x_tile()')==4
    source=source.replace('q8_0_vnni_x_tile()', 'q8_0_vnni_x_tile(n)')
    source=source.replace('static int q8_0_vnni_x_tile(n) {','static int q8_0_vnni_x_tile(int n) {',1)
    marker='void ggml_gemv_q8_0_8x8_q8_0('
    assert source.count(marker)==1
    source=source.replace(marker, '#if defined(__AVX512F__) && defined(__AVX512BW__)\n#include "qwen-q8-r8-k160-0910.h"\n#endif\n\n'+marker,1)
    start=source.index(marker);end=source.index('\ntemplate <int X_TILE>',start)
    body=source[start:end]
    marker='#if defined(__AVX512F__) && defined(__AVX512BW__)\n'
    assert body.count(marker)==1
    body=body.replace(marker,marker+'''    static const bool prepare_k160=[] {
        const char * value=std::getenv("GGML_CPU_Q8_R8_K160_PREP");
        return value && std::strcmp(value,"1")==0;
    }();
    if (prepare_k160 && n==160 && nr==1 && nc>0 && nc%8==0) {
        qwen_q8_r8_k160_prepared(s,vx,vy,nc);
        return;
    }
''',1)
    return source[:start]+body+source[end:]
