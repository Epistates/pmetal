use pmetal_runtime::InlineArray;
use pmetal_runtime::inline_array::graph_desc_count;

#[test]
fn conv1d_graph_nodes() {
    let dt = 11;
    let cd = 6144i32;
    let ck = 4i32;

    // Depthwise conv1d (groups = channels)
    let input = InlineArray::ones(&[1, ck, cd], dt);
    let weight = InlineArray::ones(&[cd, ck, 1], dt);
    let output = input.conv1d(&weight, 1, 0, 1, cd);

    let desc_count = graph_desc_count(&output);
    eprintln!("conv1d depthwise (groups={cd}): {desc_count} graph descs");

    // Also check with smaller groups
    let input2 = InlineArray::ones(&[1, ck, 128], dt);
    let weight2 = InlineArray::ones(&[128, ck, 1], dt);
    let output2 = input2.conv1d(&weight2, 1, 0, 1, 128);
    let desc2 = graph_desc_count(&output2);
    eprintln!("conv1d depthwise (groups=128): {desc2} graph descs");

    // Standard conv (groups=1)
    let input3 = InlineArray::ones(&[1, ck, 128], dt);
    let weight3 = InlineArray::ones(&[128, ck, 128], dt);
    let output3 = input3.conv1d(&weight3, 1, 0, 1, 1);
    let desc3 = graph_desc_count(&output3);
    eprintln!("conv1d standard (groups=1): {desc3} graph descs");
}
