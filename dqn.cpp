#include "dqn.hpp"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sstream>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <glog/logging.h>
#include "prettyprint.hpp"

using std::string;
using std::ios;

namespace dqn
{

DEFINE_int32(loss_display_iter, 5000, "Frequency of loss display");

void SaveInputFrame(const dqn::FrameData& frame, const std::string filename)
{
    std::ofstream ofs;
    ofs.open(filename, ios::out | ios::binary);
    for (int i = 0; i < dqn::kCroppedFrameDataSize; ++i)
    {
        ofs.write((char*) &frame[i], sizeof(uint8_t));
    }
    ofs.close();
}

/**
 * Convert pixel_t (NTSC) to RGB values.
 * Each value range [0,255]
 */
const std::array<int, 3> PixelToRGB(const pixel_t& pixel)
{
    constexpr int ntsc_to_rgb[] =
    {
        0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
        0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
        0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
        0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
        0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
        0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
        0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
        0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
        0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
        0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
        0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
        0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
        0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
        0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
        0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
        0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
        0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
        0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
        0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
        0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
        0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
        0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
        0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
        0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
        0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
        0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
        0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
        0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
        0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
        0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
        0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
        0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
    };
    const auto rgb = ntsc_to_rgb[pixel];
    const auto r = rgb >> 16;
    const auto g = (rgb >> 8) & 0xFF;
    const auto b = rgb & 0xFF;
    std::array<int, 3> arr = {r, g, b};
    return arr;
}

/**
 * Convert RGB values to a grayscale value [0,255].
 */
uint8_t RGBToGrayscale(const std::array<int, 3>& rgb)
{
    CHECK(rgb[0] >= 0 && rgb[0] <= 255);
    CHECK(rgb[1] >= 0 && rgb[1] <= 255);
    CHECK(rgb[2] >= 0 && rgb[2] <= 255);
    // Normalized luminosity grayscale
    return rgb[0] * 0.21 + rgb[1] * 0.72 + rgb[2] * 0.07;
}

uint8_t PixelToGrayscale(const pixel_t pixel)
{
    return RGBToGrayscale(PixelToRGB(pixel));
}

FrameDataSp PreprocessScreen(const ALEScreen& raw_screen)//*******************************
{
    const int raw_screen_width = raw_screen.width();
    const int raw_screen_height = raw_screen.height();
    CHECK_GT(raw_screen_height, raw_screen_width);
    const auto raw_pixels = raw_screen.getArray();
    auto screen = std::make_shared<FrameData>();
    // Crop the top of the screen
    const int cropped_screen_height = static_cast<int>(.85 * raw_screen_height);
    const int start_y = raw_screen_height - cropped_screen_height;
    // Ignore the leftmost column of 8 pixels
    const int start_x = 8;
    const int cropped_screen_width = raw_screen_width - start_x;
    const auto x_ratio = cropped_screen_width / static_cast<double>(kCroppedFrameSize);
    const auto y_ratio = cropped_screen_height / static_cast<double>(kCroppedFrameSize);
    for (auto i = 0; i < kCroppedFrameSize; ++i)
    {
        for (auto j = 0; j < kCroppedFrameSize; ++j)
        {
            const auto first_x = start_x + static_cast<int>(std::floor(j * x_ratio));
            const auto last_x = start_x + static_cast<int>(std::floor((j + 1) * x_ratio));
            const auto first_y = start_y + static_cast<int>(std::floor(i * y_ratio));
            const auto last_y = start_y + static_cast<int>(std::floor((i + 1) * y_ratio));
            uint8_t resulting_color = 0.0;
            for (auto x = first_x; x <= last_x; ++x)
            {
                double x_ratio_in_resulting_pixel = 1.0;
                if (x == first_x)
                {
                    x_ratio_in_resulting_pixel = x + 1 - j * x_ratio - start_x;
                }
                else if (x == last_x)
                {
                    x_ratio_in_resulting_pixel = x_ratio * (j + 1) - x + start_x;
                }
                assert(x_ratio_in_resulting_pixel >= 0.0 &&
                       x_ratio_in_resulting_pixel <= 1.0);
                for (auto y = first_y; y <= last_y; ++y)
                {
                    double y_ratio_in_resulting_pixel = 1.0;
                    if (y == first_y)
                    {
                        y_ratio_in_resulting_pixel = y + 1 - i * y_ratio - start_y;
                    }
                    else if (y == last_y)
                    {
                        y_ratio_in_resulting_pixel = y_ratio * (i + 1) - y + start_y;
                    }
                    assert(y_ratio_in_resulting_pixel >= 0.0 &&
                           y_ratio_in_resulting_pixel <= 1.0);
                    const auto grayscale =
                        PixelToGrayscale(
                            raw_pixels[static_cast<int>(y * raw_screen_width + x)]);
                    resulting_color +=
                        (x_ratio_in_resulting_pixel / x_ratio) *
                        (y_ratio_in_resulting_pixel / y_ratio) * grayscale;
                }
            }
            (*screen)[i * kCroppedFrameSize + j] = resulting_color;
        }
    }
    return screen;
}

void DQN::ObscureScreen(FrameDataSp& screen, double obscure_prob)//*******************************
{
    CHECK(obscure_prob >= 0 && obscure_prob <= 1);
    if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < obscure_prob)
    {
        std::fill(screen->begin(), screen->end(), 0);
    }
}

void DQN::RedisplayScreen(FrameDataSp& screen, double prob)//*******************************
{
    CHECK(prob >= 0 && prob <= 1);
    if (last_displayed_screen_.empty())
    {
        last_displayed_screen_.resize(kCroppedFrameDataSize);
        std::copy(screen->begin(), screen->end(), last_displayed_screen_.begin());
    }
    if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < prob)
    {
        std::copy(last_displayed_screen_.begin(), last_displayed_screen_.end(),
                  screen->begin());
    }
    else
    {
        std::copy(screen->begin(), screen->end(), last_displayed_screen_.begin());
    }
}


void PrintQValues(const std::vector<float>& q_values, const ActionVect& actions)
{
    CHECK_GT(q_values.size(), 0);
    CHECK_GT(actions.size(), 0);
    CHECK_EQ(q_values.size(), actions.size());
    std::ostringstream actions_buf;
    std::ostringstream q_values_buf;
    for (auto i = 0; i < q_values.size(); ++i)
    {
        const auto a_str =
            boost::algorithm::replace_all_copy(
                action_to_string(actions[i]), "PLAYER_A_", "");
        const auto q_str = std::to_string(q_values[i]);
        const auto column_size = std::max(a_str.size(), q_str.size()) + 1;
        actions_buf.width(column_size);
        actions_buf << a_str;
        q_values_buf.width(column_size);
        q_values_buf << q_str;
    }
    LOG(INFO) << actions_buf.str();
    LOG(INFO) << q_values_buf.str();
}


template <typename Dtype>
void HasBlobSize(caffe::Net<Dtype>& net,
                 const std::string& blob_name,
                 const std::vector<int> expected_shape)
{
    net.has_blob(blob_name);
    const caffe::Blob<Dtype>& blob = *net.blob_by_name(blob_name);
    const std::vector<int>& blob_shape = blob.shape();
    CHECK_EQ(blob_shape.size(), expected_shape.size());
    CHECK(std::equal(blob_shape.begin(), blob_shape.end(),
                     expected_shape.begin()));
}

void PopulateLayer(caffe::LayerParameter& layer,
                   const std::string& name,
                   const std::string& type,
                   const std::vector<std::string>& bottoms,
                   const std::vector<std::string>& tops,
                   const boost::optional<caffe::Phase>& include_phase)
{
    layer.set_name(name);
    layer.set_type(type);
    for (auto& bottom : bottoms)
    {
        layer.add_bottom(bottom);
    }
    for (auto& top : tops)
    {
        layer.add_top(top);
    }
    // PopulateLayer(layer, name, type, bottoms, tops);
    if (include_phase)
    {
        layer.add_include()->set_phase(*include_phase);
    }
}


void MemoryDataLayer(caffe::NetParameter& net_param,
                     const std::string& name,
                     const std::vector<std::string>& tops,
                     const boost::optional<caffe::Phase>& include_phase,
                     const std::vector<int>& shape)
{
    caffe::LayerParameter& memory_layer = *net_param.add_layer();
    PopulateLayer(memory_layer, name, "MemoryData", {}, tops, include_phase);
    CHECK_EQ(shape.size(), 4);
    caffe::MemoryDataParameter* memory_data_param = memory_layer.mutable_memory_data_param();
    memory_data_param->set_batch_size(shape[0]);
    memory_data_param->set_channels(shape[1]);
    memory_data_param->set_height(shape[2]);
    memory_data_param->set_width(shape[3]);
}


void ReshapeLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase,
                  const std::vector<int>& shape)
{
    caffe::LayerParameter& reshape_layer = *net_param.add_layer();
    PopulateLayer(reshape_layer, name, "Reshape", bottoms, tops, include_phase);
    caffe::ReshapeParameter* reshape_param = reshape_layer.mutable_reshape_param();
    caffe::BlobShape* blob_shape = reshape_param->mutable_shape();
    for (auto& dim : shape)
    {
        blob_shape->add_dim(dim);
    }
}

void SliceLayer(caffe::NetParameter& net_param,
                const std::string& name,
                const std::vector<std::string>& bottoms,
                const std::vector<std::string>& tops,
                const boost::optional<caffe::Phase>& include_phase,
                const int axis,
                const std::vector<int>& slice_points)
{
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Slice", bottoms, tops, include_phase);
    caffe::SliceParameter* slice_param = layer.mutable_slice_param();
    slice_param->set_axis(axis);
    for (auto& p : slice_points)
    {
        slice_param->add_slice_point(p);
    }
}

void ConvLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const std::string& shared_name,
               const float& lr_mult,
               const boost::optional<caffe::Phase>& include_phase,
               const int num_output,
               const int kernel_size,
               const int stride)
{//ConvLayer(np, "conv1", {"all_frames"}, {"conv1"}, "conv1", -1,boost::none, 32, 8, 4);
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Convolution", bottoms, tops, include_phase);
    caffe::ParamSpec* weight_param = layer.add_param();
    weight_param->set_name(shared_name + "_w");
    if (lr_mult >= 0)
    {
        weight_param->set_lr_mult(lr_mult);
    }
    weight_param->set_decay_mult(1);
    caffe::ParamSpec* bias_param = layer.add_param();
    bias_param->set_name(shared_name + "_b");
    if (lr_mult >= 0)
    {
        bias_param->set_lr_mult(2 * lr_mult);
    }
    bias_param->set_decay_mult(0);
    caffe::ConvolutionParameter* conv_param = layer.mutable_convolution_param();
    conv_param->set_num_output(num_output);
    conv_param->set_kernel_size(kernel_size);
    conv_param->set_stride(stride);
    caffe::FillerParameter* weight_filler = conv_param->mutable_weight_filler();
    weight_filler->set_type("gaussian");
    weight_filler->set_std(0.01);
    caffe::FillerParameter* bias_filler = conv_param->mutable_bias_filler();
    bias_filler->set_type("constant");
    bias_filler->set_value(0);
}

void ReluLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const boost::optional<caffe::Phase>& include_phase)
{
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "ReLU", bottoms, tops, include_phase);
    caffe::ReLUParameter* relu_param = layer.mutable_relu_param();
    relu_param->set_negative_slope(0.01);
}

void IPLayer(caffe::NetParameter& net_param,
             const std::string& name,
             const std::vector<std::string>& bottoms,
             const std::vector<std::string>& tops,
             const std::string& shared_name,
             const float& lr_mult,
             const boost::optional<caffe::Phase>& include_phase,
             const int num_output,
             const int axis)
{
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "InnerProduct", bottoms, tops, include_phase);
    caffe::ParamSpec* weight_param = layer.add_param();
    weight_param->set_name(shared_name + "_w");
    if (lr_mult >= 0)
    {
        weight_param->set_lr_mult(lr_mult);
    }
    weight_param->set_decay_mult(1);
    caffe::ParamSpec* bias_param = layer.add_param();
    bias_param->set_name(shared_name + "_b");
    if (lr_mult >= 0)
    {
        bias_param->set_lr_mult(2 * lr_mult);
    }
    bias_param->set_decay_mult(0);
    caffe::InnerProductParameter* ip_param = layer.mutable_inner_product_param();
    ip_param->set_num_output(num_output);
    ip_param->set_axis(axis);
    caffe::FillerParameter* weight_filler = ip_param->mutable_weight_filler();
    weight_filler->set_type("gaussian");
    weight_filler->set_std(0.005);
    caffe::FillerParameter* bias_filler = ip_param->mutable_bias_filler();
    bias_filler->set_type("constant");
    bias_filler->set_value(1);
}

void ConcatLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase,
                 const int& axis)
{
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Concat", bottoms, tops, include_phase);
    caffe::ConcatParameter* concat_param = layer.mutable_concat_param();
    concat_param->set_axis(axis);
}

void LstmLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const boost::optional<caffe::Phase>& include_phase,
               const int& num_output)
{
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "LSTM", bottoms, tops, include_phase);
    caffe::RecurrentParameter* recurrent_param = layer.mutable_recurrent_param();
    recurrent_param->set_num_output(num_output);
    caffe::FillerParameter* weight_filler = recurrent_param->mutable_weight_filler();
    weight_filler->set_type("uniform");
    weight_filler->set_min(-0.08);
    weight_filler->set_max(0.08);
    caffe::FillerParameter* bias_filler = recurrent_param->mutable_bias_filler();
    bias_filler->set_type("constant");
    bias_filler->set_value(0);
}

void EltwiseLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase,
                  const caffe::EltwiseParameter::EltwiseOp& op)
{
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Eltwise", bottoms, tops, include_phase);
    caffe::EltwiseParameter* eltwise_param = layer.mutable_eltwise_param();
    eltwise_param->set_operation(op);
}

//prevent top-level blobs from being printed during training
void SilenceLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase)
{
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Silence", bottoms, tops, include_phase);
}

void EuclideanLossLayer(caffe::NetParameter& net_param,
                        const std::string& name,
                        const std::vector<std::string>& bottoms,
                        const std::vector<std::string>& tops,
                        const boost::optional<caffe::Phase>& include_phase)
{
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "EuclideanLoss", bottoms, tops, include_phase);
}

caffe::NetParameter DQN::CreateNet(bool unroll1_is_lstm) //*******************************
{
    caffe::NetParameter np;
    np.set_name("Deep Recurrent Q-Network");

    MemoryDataLayer(np, frames_layer_name, {train_frames_blob_name,"dummy_frames"}, caffe::TRAIN,{minibatch_size_, frames_per_forward_, kCroppedFrameSize, kCroppedFrameSize});
    MemoryDataLayer(np, cont_layer_name, {cont_blob_name,"dummy_cont"}, caffe::TRAIN,{unroll_, minibatch_size_, 1, 1});
    MemoryDataLayer(np, target_layer_name, {target_blob_name,"dummy_target"}, caffe::TRAIN,{unroll_, minibatch_size_, kOutputCount, 1});
    MemoryDataLayer(np, filter_layer_name, {filter_blob_name,"dummy_filter"}, caffe::TRAIN,{unroll_, minibatch_size_, kOutputCount, 1});
    MemoryDataLayer(np, last_action_layer_name, {train_last_action_blob_name,"dummy_last_actions"}, caffe::TRAIN,{minibatch_size_, frames_per_forward_, kOutputCount, 1});

    SilenceLayer(np, "silence", {"dummy_frames","dummy_cont","dummy_filter","dummy_target","dummy_last_actions"}, {}, caffe::TRAIN);

    ReshapeLayer(np, "reshape_cont", {cont_blob_name}, {"reshaped_cont"}, caffe::TRAIN,{unroll_, minibatch_size_});
    ReshapeLayer(np, "reshape_filter", {filter_blob_name}, {"reshaped_filter"}, caffe::TRAIN,{unroll_, minibatch_size_, kOutputCount});

    MemoryDataLayer(np, frames_layer_name, {test_frames_blob_name,"dummy_frames"},caffe::TEST,{minibatch_size_,frames_per_timestep_,kCroppedFrameSize,kCroppedFrameSize});
    MemoryDataLayer(np, cont_layer_name, {cont_blob_name,"dummy_cont"}, caffe::TEST,{1, minibatch_size_, 1, 1});
    MemoryDataLayer(np, last_action_layer_name, {test_last_action_blob_name,"dummy_last_actions"}, caffe::TEST,{minibatch_size_, frames_per_timestep_, kOutputCount, 1});

    SilenceLayer(np, "silence", {"dummy_frames","dummy_cont","dummy_last_actions"}, {}, caffe::TEST);

    ReshapeLayer(np, "reshape_cont", {cont_blob_name}, {"reshaped_cont"}, caffe::TEST,{1, minibatch_size_});

    if (unroll_ > 1)
    {

        std::vector<std::string> frames_tops, scrap_tops, actions_tops, action_scrap_tops;
        for (int t = 0; t < unroll_; ++t)
        {
            std::string ts = std::to_string(t);
            boost::optional<caffe::Phase> phase;
            if (t > 0)
            {
                phase.reset(caffe::TRAIN);
            }
            std::vector<int> slice_points, Action_slice_points;
            std::vector<std::string> slice_tops, Action_slice_tops;
            if (t == 0)
            {
                slice_points = {frames_per_timestep_};
                slice_tops = {"frames_"+ts, "scrap_"+ts};
                scrap_tops.push_back("scrap_"+ts);

				Action_slice_points = {frames_per_timestep_};
                Action_slice_tops = {"actions_"+ts, "action_scrap_"+ts};
                action_scrap_tops.push_back("action_scrap_"+ts);
            }
            else if (t == unroll_ - 1)
            {
                slice_points = {t};
                slice_tops = {"scrap_"+ts, "frames_"+ts};
                scrap_tops.push_back("scrap_"+ts);

                Action_slice_points = {t};
                Action_slice_tops = {"action_scrap_"+ts,"actions_"+ts};
                action_scrap_tops.push_back("action_scrap_"+ts);
            }
            else
            {
				slice_tops = {"scrap1_"+ts, "frames_"+ts, "scrap2_"+ts};
                scrap_tops.push_back("scrap1_"+ts);
                scrap_tops.push_back("scrap2_"+ts);
                slice_points = {t, t + frames_per_timestep_};

				Action_slice_tops = {"action_scrap1_"+ts, "actions_"+ts, "action_scrap2_"+ts};
                action_scrap_tops.push_back("action_scrap1_"+ts);
                action_scrap_tops.push_back("action_scrap2_"+ts);
                Action_slice_points = {t, t + frames_per_timestep_};

            }
            SliceLayer(np, "slice_"+ts, {train_frames_blob_name}, slice_tops,caffe::TRAIN, 1, slice_points);
            SliceLayer(np, "action_slice_"+ts, {train_last_action_blob_name}, Action_slice_tops, caffe::TRAIN, 1, Action_slice_points);

			frames_tops.push_back("frames_"+ts);
			actions_tops.push_back("actions_"+ts);
        }
        SilenceLayer(np, "scrap_silence", scrap_tops, {}, caffe::TRAIN);
        SilenceLayer(np, "action_scrap_silence", action_scrap_tops, {}, caffe::TRAIN);
        ConcatLayer(np, "concat_frames", frames_tops, {"all_frames"}, caffe::TRAIN,0);
        ConcatLayer(np, "concat_actions", actions_tops, {test_last_action_blob_name}, caffe::TRAIN,0);//test_last_action_blob_name = "last_actions"
        ConvLayer(np, "conv1", {"all_frames"}, {"conv1"}, "conv1", -1,boost::none, 32, 8, 4);
    }
    else
    {
        ConvLayer(np, "conv1", {train_frames_blob_name}, {"conv1"}, "conv1", -1,caffe::TRAIN, 32, 8, 4);
        ConvLayer(np, "conv1", {"all_frames"}, {"conv1"}, "conv1", -1, caffe::TEST,32, 8, 4);
    }

    ReluLayer(np, "conv1_relu", {"conv1"}, {"conv1"}, boost::none);
    ConvLayer(np, "conv2", {"conv1"}, {"conv2"}, "conv2", -1, boost::none,64, 4, 2);

    ReluLayer(np, "conv2_relu", {"conv2"}, {"conv2"}, boost::none);
    ConvLayer(np, "conv3", {"conv2"}, {"conv3"}, "conv3", -1, boost::none,64, 3, 1);

    ReluLayer(np, "conv3_relu", {"conv3"}, {"conv3"}, boost::none);
    ReshapeLayer(np, "conv3_reshape", {"conv3"}, {"reshaped_conv3"},caffe::TRAIN, {unroll_, minibatch_size_, 64*7*7});
    ReshapeLayer(np, "conv3_reshape", {"conv3"}, {"reshaped_conv3"},caffe::TEST, {1, minibatch_size_, 64*7*7});

    ReshapeLayer(np, "reshape_last_action", {test_last_action_blob_name}, {"reshaped_last_action"}, caffe::TRAIN,{unroll_, minibatch_size_, frames_per_timestep_ * kOutputCount});///dim: 10 32 4
    ReshapeLayer(np, "reshape_last_action", {test_last_action_blob_name}, {"reshaped_last_action"}, caffe::TEST,{1, minibatch_size_,frames_per_timestep_ * kOutputCount});///dim: 1 32 4
    IPLayer(np,"ip_action",{"reshaped_last_action"},{"action_embedding"},"ip_action",-1,boost::none,lstmSize, 2);
	ConcatLayer(np, "concat_all", {"action_embedding", "reshaped_conv3"}, {"concat_all"}, boost::none,2);//test_last_action_blob_name = "last_actions"

    if (unroll_ > 1 || unroll1_is_lstm)
    {
        LstmLayer(np, "lstm1", {"concat_all","reshaped_cont",}, {"lstm1"}, boost::none,lstmSize);//constexpr auto lstmSize = 512;
    }
    else
    {
        IPLayer(np, "lstm1", {"concat_all"}, {"lstm1"}, "lstm1", -1, boost::none,lstmSize, 2);
        ReluLayer(np, "ip1_relu", {"lstm1"}, {"lstm1"}, boost::none);
        SilenceLayer(np, "cont_silence", {"reshaped_cont"}, {}, boost::none);
    }

    IPLayer(np, "ip2", {"lstm1"}, {q_values_blob_name}, "ip2", -1, boost::none,kOutputCount, 2);
    EltwiseLayer(np, "eltwise_filter", {q_values_blob_name,"reshaped_filter"},{"filtered_q_values"}, caffe::TRAIN, caffe::EltwiseParameter::PROD);
    EuclideanLossLayer(np, "loss", {"filtered_q_values","target"}, {"loss"},caffe::TRAIN);
    return np;
}

int ParseIterFromSnapshot(const std::string& snapshot)
{
    unsigned start = snapshot.find_last_of("_");
    unsigned end = snapshot.find_last_of(".");
    return std::stoi(snapshot.substr(start+1, end-start-1));
}

float ParseScoreFromSnapshot(const std::string& snapshot)
{
    unsigned start = snapshot.find("_HiScore_");
    unsigned end = snapshot.find("_std_");
    return std::stof(snapshot.substr(start+9, end-start-9));
}

void RemoveSnapshots(const std::string& snapshot_prefix, int min_iter)
{
    std::string regexp(snapshot_prefix +
                       "_iter_[0-9]+\\.(caffemodel|solverstate|replaymemory)");
    for (const std::string& f : FilesMatchingRegexp(regexp))
    {
        int iter = ParseIterFromSnapshot(f);
        if (iter < min_iter)
        {
            LOG(INFO) << "Removing " << f;
            CHECK(boost::filesystem::is_regular_file(f));
            boost::filesystem::remove(f);
        }
    }
}

std::string FindLatestSnapshot(const std::string& snapshot_prefix)
{
    using namespace boost::filesystem;
    std::string regexp(snapshot_prefix + "_iter_[0-9]+\\.solverstate");
    std::vector<std::string> matching_files = FilesMatchingRegexp(regexp);
    int max_iter = -1;
    std::string latest = "";
    for (const std::string& f : matching_files)
    {
        int iter = ParseIterFromSnapshot(f);
        if (iter > max_iter)
        {
            // Look for an associated caffemodel + replaymemory
            path p(f);
            p = p.parent_path() / p.stem();
            std::string caffemodel = p.native() + ".caffemodel";
            std::string replaymemory = p.native() + ".replaymemory";
            if (is_regular_file(caffemodel) && is_regular_file(replaymemory))
            {
                max_iter = iter;
                latest = f;
            }
        }
    }
    return latest;
}

std::vector<std::string> GetHiScoreSnapshots(const std::string& snapshot_prefix)
{
    std::string regexp(snapshot_prefix + "_HiScore.*_iter_[0-9]+\\.(caffemodel|solverstate)");
    return FilesMatchingRegexp(regexp);
}

float FindHiScore(const std::string& snapshot_prefix)
{
    using namespace boost::filesystem;
    float max_score = std::numeric_limits<float>::lowest();
    for (const std::string& f : GetHiScoreSnapshots(snapshot_prefix))
    {
        float score = ParseScoreFromSnapshot(f);
        if (score > max_score)
        {
            max_score = score;
        }
    }
    return max_score;
}

DQN::DQN(const ActionVect& legal_actions,
         const int replay_memory_capacity,
         const double gamma,
         const int clone_frequency,
         const int unroll,
         const int minibatch_size,
         const int frames_per_timestep) :
    legal_actions_(legal_actions),
    replay_memory_capacity_(replay_memory_capacity),
    gamma_(gamma),
    clone_frequency_(clone_frequency),
    unroll_(unroll),
    frames_per_timestep_(frames_per_timestep),
    minibatch_size_(minibatch_size),
    random_engine(0),
    smoothed_loss_(0)
{
    last_clone_iter_ = 0;
    replay_memory_size_ = 0;
    frames_per_forward_ = unroll_ + frames_per_timestep_ - 1;
    frame_input_size_TRAIN_ = minibatch_size_ * frames_per_forward_ * kCroppedFrameDataSize;
	last_action_input_size_TRAIN_ = minibatch_size_ * frames_per_forward_ * kOutputCount;
    target_input_size_TRAIN_ = unroll_ * minibatch_size_ * kOutputCount;
    filter_input_size_TRAIN_ = unroll_ * minibatch_size_ * kOutputCount;
    cont_input_size_TRAIN_ = unroll_ * minibatch_size_;


    frame_input_size_TEST_ = minibatch_size_ * frames_per_timestep_ *
                             kCroppedFrameDataSize;
    cont_input_size_TEST_ = minibatch_size_;
	last_action_input_size_TEST_ = minibatch_size_* frames_per_timestep_ * kOutputCount;// * 1 * 1
}


void DQN::LoadTrainedModel(const std::string& model_bin)
{
    net_->CopyTrainedLayersFrom(model_bin);
}

void DQN::RestoreSolver(const std::string& solver_bin)
{
    solver_->Restore(solver_bin.c_str());
}

std::vector<std::string> FilesMatchingRegexp(const std::string& regexp)
{
    using namespace boost::filesystem;
    path search_stem(regexp);
    path search_dir(current_path());
    if (search_stem.has_parent_path())
    {
        search_dir = search_stem.parent_path();
        search_stem = search_stem.filename();
    }
    const boost::regex expression(search_stem.native());
    std::vector<std::string> matching_files;
    directory_iterator end;
    for(directory_iterator it(search_dir); it != end; ++it)
    {
        if (is_regular_file(it->status()))
        {
            path p(it->path());
            boost::smatch what;
            if (boost::regex_match(p.filename().native(), what, expression))
            {
                matching_files.push_back(p.native());
            }
        }
    }
    return matching_files;
}

void DQN::SnapshotHiScore(const std::string& snapshot_prefix,
                          double avg_score, double std_dev,
                          bool remove_old)
{
    std::stringstream out;
    out.precision(1);
    out << std::fixed << snapshot_prefix << "_HiScore_"  << avg_score << "_std_" << std_dev;
    std::string high_score_prefix = out.str();
    Snapshot(high_score_prefix, false, false);
    if (remove_old)
    {
        for (const std::string f : GetHiScoreSnapshots(snapshot_prefix))
        {
            if (f.substr(0, high_score_prefix.size()) != high_score_prefix)
            {
                LOG(INFO) << "Removing " << f;
                CHECK(boost::filesystem::is_regular_file(f));
                boost::filesystem::remove(f);
            }
        }
    }
}

void DQN::Snapshot(const std::string& snapshot_prefix, bool remove_old,
                   bool snapshot_memory)
{
    using namespace boost::filesystem;
    solver_->Snapshot(snapshot_prefix);
    int snapshot_iter = current_iteration() + 1;
    std::string fname = snapshot_prefix + "_iter_" + std::to_string(snapshot_iter);
    CHECK(is_regular_file(fname + ".caffemodel"));
    CHECK(is_regular_file(fname + ".solverstate"));
    if (snapshot_memory)
    {
        std::string mem_fname = fname + ".replaymemory";
        LOG(INFO) << "Snapshotting memory to " << mem_fname;
        SnapshotReplayMemory(mem_fname);
        CHECK(is_regular_file(mem_fname));
    }
    if (remove_old)
    {
        RemoveSnapshots(snapshot_prefix, snapshot_iter);
    }
}

void DQN::Initialize(caffe::SolverParameter& solver_param)
{
    // Initialize net and solver
    solver_.reset(caffe::GetSolver<float>(solver_param));
    net_ = solver_->net();
    CHECK_EQ(solver_->test_nets().size(), 1);
    test_net_ = solver_->test_nets()[0];
    // Test Net shares parameters with train net at all times
    test_net_->ShareTrainedLayersWith(net_.get());
    // Clone net maintains its own set of parameters
    CloneNet(*test_net_);
    // Check the primary network
    HasBlobSize(*net_, train_frames_blob_name, {minibatch_size_,frames_per_forward_, kCroppedFrameSize, kCroppedFrameSize});
    HasBlobSize(*net_, target_blob_name, {unroll_,minibatch_size_,kOutputCount,1});
    HasBlobSize(*net_, filter_blob_name, {unroll_,minibatch_size_,kOutputCount,1});
    HasBlobSize(*net_, cont_blob_name, {unroll_, minibatch_size_, 1, 1});
    // Check the test network
    HasBlobSize(*test_net_, test_frames_blob_name, {minibatch_size_,frames_per_timestep_, kCroppedFrameSize, kCroppedFrameSize});
    HasBlobSize(*test_net_, cont_blob_name, {1, minibatch_size_, 1, 1});
    LOG(INFO) << "Finished " << net_->name() << " Initialization";
}

Action DQN::SelectAction(const InputFrames& frames,
                         const LastAction& last_action,
                         const double epsilon,
                         bool cont)//*******************************
{
    return SelectActions(InputFramesBatch{{frames}}, LastActionBatch{{last_action}}, epsilon, cont)[0];
}

ActionVect DQN::SelectActions(const InputFramesBatch& frames_batch,
                              const LastActionBatch& last_action_batch,
                              const double epsilon,
                              bool cont)
{
    CHECK(epsilon <= 1.0 && epsilon >= 0.0);
    CHECK_EQ(frames_batch.size(), last_action_batch.size());
    CHECK_LE(frames_batch.size(), minibatch_size_);

    ActionVect actions(frames_batch.size());
    if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < epsilon)
    {
        // Select randomly
        for (int i = 0; i < actions.size(); ++i)
        {
            const auto random_idx = std::uniform_int_distribution<int>(0, legal_actions_.size() - 1)(random_engine);
            actions[i] = legal_actions_[random_idx];
        }
    }
    else
    {
        // Select greedily
        std::vector<ActionValue> actions_and_values = SelectActionGreedily(*test_net_, frames_batch, last_action_batch, cont);
        CHECK_EQ(actions_and_values.size(), actions.size());
        for (int i=0; i<actions_and_values.size(); ++i)
        {
            actions[i] = actions_and_values[i].first;//ActionValue --- std::pair<Action, float>
        }
    }
    return actions;
}

ActionValue DQN::SelectActionGreedily(caffe::Net<float>& net,
                                      const InputFrames& last_frames,
                                      const Action& last_action,
                                      bool cont)
{
    return SelectActionGreedily(net, InputFramesBatch{{last_frames}}, LastActionBatch{{last_action}}, cont).front();
}

std::vector<ActionValue> DQN::SelectActionGreedily(caffe::Net<float>& net,
												   const InputFramesBatch& frames_batch,
                                                   const LastActionBatch& last_action_batch,
                                                   bool cont)
{
    std::vector<ActionValue> results;
    if(frames_batch.empty() || last_action_batch.empty())
    {
        return results;
    }
    CHECK_EQ(net.phase(), caffe::TEST);
    CHECK(net.has_blob(test_frames_blob_name));
    const auto frames_blob = net.blob_by_name(test_frames_blob_name);//test_frames_blob_name = "all_frames"
	const auto last_action_blob = net.blob_by_name(test_last_action_blob_name);

    CHECK_LE(frames_batch.size(), minibatch_size_);
    std::vector<float> frame_input(frame_input_size_TEST_, 0.0f);
    std::vector<float> cont_input(cont_input_size_TEST_, cont);
	std::vector<float> last_action_input(last_action_input_size_TEST_, 0.0f);
    // Input frames to the net and compute Q values for each legal action
    for (int n = 0; n < frames_batch.size(); ++n)
    {
        const InputFrames& input_frames = frames_batch[n];
		const LastAction input_actions = last_action_batch[n];


        for (int i = 0; i < frames_per_timestep_; ++i)
        {
            const FrameDataSp& frame_data = input_frames[i];
			const int action_data = input_actions[i];

            std::copy(frame_data->begin(), frame_data->end(),
                      frame_input.begin() + frames_blob->offset(n,i,0,0));

			last_action_input[last_action_blob->offset(n,i,action_data,0)] =1.0f;
        }
    }

    InputDataIntoLayers(net, frame_input.data(), last_action_input.data(), cont_input.data(), NULL, NULL);
    net.ForwardPrefilled(nullptr);
    // Collect the Results
    results.reserve(frames_batch.size());
    CHECK(net.has_blob(q_values_blob_name));
    const auto q_values_blob = net.blob_by_name(q_values_blob_name);
    for (int i = 0; i < frames_batch.size(); ++i)
    {
        // Get the Q-values from the net: unroll_*minibatch_size_*kOutputCount
        const auto action_evaluator = [&](Action action)
        {
            const auto q = q_values_blob->data_at(0, i, static_cast<int>(action), 0);
            CHECK(!std::isnan(q));
            return q;
        };
        std::vector<float> q_values(legal_actions_.size());
        std::transform(legal_actions_.begin(), legal_actions_.end(),
                       q_values.begin(), action_evaluator);
        // PrintQValues(q_values, legal_actions_);
        // Select the action with the maximum Q value
        const auto max_idx = std::distance(
                                 q_values.begin(),
                                 std::max_element(q_values.begin(), q_values.end()));
        results.emplace_back(legal_actions_[max_idx], q_values[max_idx]);
    }
    return results;
}

void DQN::RememberEpisode(const Episode& episode)//*******************************
{
    replay_memory_size_ += episode.size();
    replay_memory_.push_back(episode);
    while (replay_memory_size_ >= replay_memory_capacity_)
    {
        replay_memory_size_ -= replay_memory_.front().size();
        replay_memory_.pop_front();
    }
}
/**
int DQN::UpdateSequential()//***********while********************
{
    // Every clone_iters steps, update the clone_net_
    if (!clone_net_ || current_iteration() >= last_clone_iter_ + clone_frequency_)
    {
        LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
        CloneNet(*test_net_);
        last_clone_iter_ = current_iteration();
    }
    const auto frames_blob = net_->blob_by_name(train_frames_blob_name);
    const auto cont_blob   = net_->blob_by_name(cont_blob_name);
    const auto filter_blob = net_->blob_by_name(filter_blob_name);
    const auto target_blob = net_->blob_by_name(target_blob_name);
    // Randomly select unique episodes to learn from
    std::vector<int> ep_inds(replay_memory_.size());
    std::iota(ep_inds.begin(), ep_inds.end(), 0);
    auto func = [this](int i)
    {
        return random_engine() % i;
    };
    std::random_shuffle(ep_inds.begin(), ep_inds.end(), func);
    if (ep_inds.size() > minibatch_size_)
    {
        ep_inds.resize(minibatch_size_);
    }
    int max_episode_length = 0;
    for (auto i : ep_inds)
    {
        if (replay_memory_[i].size() > max_episode_length)
        {
            max_episode_length = replay_memory_[i].size();
        }
    }
    int t = frames_per_timestep_-1; // Timestep to start at
    const int n_updates = int(ceil((max_episode_length-t)/float(unroll_)));
    const int batch = ep_inds.size();
    for (int update = 0; update < n_updates; ++update, t += unroll_)
    {
        InputFramesBatch frames_batch(batch);
        std::vector<float> frame_input(frame_input_size_TRAIN_, 0.0f);
        std::vector<float> filter_input(filter_input_size_TRAIN_, 0.0f);
        std::vector<float> target_input(target_input_size_TRAIN_, 0.0f);
        std::vector<float> cont_input(cont_input_size_TRAIN_, 1.0f);
        if (update == 0)   // Cont is zeroed only for the first step of the episode
        {
            for (int n = 0; n < minibatch_size_; ++n)
            {
                cont_input[cont_blob->offset(0,n,0,0)] = 0.0f;
            }
        }

        // Load the frames necessary for a full update
        for (int n = 0; n < batch; ++n)
        {
            Episode& episode = replay_memory_[ep_inds[n]];
            InputFrames& frames = frames_batch[n];
            int ts = t - (frames_per_timestep_-1);
            for (int i=0; i<=frames_per_forward_ && ts<episode.size(); ++i, ++ts)
            {
                FrameDataSp& frame = std::get<0>(episode[ts]);
                frames.push_back(frame);
                if (i < frames_per_forward_)
                {
                    std::copy(frame->begin(), frame->end(),
                              frame_input.begin() + frames_blob->offset(n,i,0,0));
                }
            }
        }
        // Generate unroll_ batches of next-state q-values
        std::vector<std::vector<ActionValue> > qvals(unroll_);
        for (int i = 0; i < unroll_; ++i)
        {
            InputFramesBatch test_frames_batch;
            for (int n = 0; n < batch; ++n)
            {
                const InputFrames& frames = frames_batch[n];
                if (frames.size() >= i + frames_per_timestep_ + 1)
                {
                    InputFrames test_frames(frames_per_timestep_);
                    for (int f = 0; f < frames_per_timestep_; ++f)
                    {
                        test_frames[f] = frames[i + f + 1];
                    }
                    test_frames_batch.emplace_back(test_frames);
                }
            }
            std::vector<ActionValue> actions_and_values =
                SelectActionGreedily(*clone_net_, test_frames_batch, update>0 || i>0);
            qvals[i] = actions_and_values;
        }
        // Generate the target/filter inputs
        for (int i = 0; i < unroll_; ++i)
        {
            int target_value_idx = 0;
            for (int n = 0; n < batch; ++n)
            {
                const Episode& episode = replay_memory_[ep_inds[n]];
                if (t + i < episode.size())
                {
                    const auto& transition = episode[t + i];
                    const int action = static_cast<int>(std::get<1>(transition));
                    CHECK_LT(action, kOutputCount);
                    const float reward = std::get<2>(transition);
                    CHECK(reward >= -1.0 && reward <= 1.0);
                    const float target = std::get<3>(transition) ?
                                         reward + gamma_ * qvals[i][target_value_idx++].second :
                                         reward;
                    CHECK(!std::isnan(target));
                    filter_input[filter_blob->offset(i,n,action,0)] = 1;
                    target_input[target_blob->offset(i,n,action,0)] = target;
                }
            }
            CHECK_EQ(target_value_idx, qvals[i].size());
        }
        InputDataIntoLayers(*net_, frame_input.data(), cont_input.data(),
                            target_input.data(), filter_input.data());
        solver_->Step(1);
    }
    return n_updates;
}
*/


void DQN::Benchmark(int iterations, bool random_updates)
{
    UpdateRandom();
    while (memory_episodes() < minibatch_size_)
    {
        RememberEpisode(replay_memory_[0]);
    }
    LOG(INFO) << "*** Benchmark begins ***";
    if (random_updates)
    {
        LOG(INFO) << "Testing Random Updates for " << iterations << " iterations.";
    }
    else
    {
        LOG(INFO) << "Testing Sequential Updates.";
    }
    caffe::Timer total_timer;
    total_timer.Start();
    int n_updates = 0;
    caffe::Timer update_timer;
    update_timer.Start();
    while (n_updates < iterations)
    {
        n_updates += UpdateRandom();
    }
    update_timer.Stop();
    LOG(INFO) << "Average Update: " << update_timer.MilliSeconds() /
              n_updates << " ms.";
    CHECK(memory_size() > frames_per_forward_);
    InputFrames frames;
    for (int i = 0; i < frames_per_forward_; ++i)
    {
        frames.push_back(std::get<0>(replay_memory_[0][i]));
    }
    caffe::Timer select_timer;
    select_timer.Start();
    for (int j = 0; j < iterations; ++j)
    {
        SelectAction(frames, 0, true);
    }
    select_timer.Stop();
    LOG(INFO) << "Average Select Action: " << select_timer.MilliSeconds() /
              iterations << " ms.";
    total_timer.Stop();
    LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
    auto h = 1000000. / iterations * total_timer.MilliSeconds() / 1000. / 3600.;
    LOG(INFO) << "Estimated Time to 1M iters: " << h << " hours.";
    LOG(INFO) << "*** Benchmark ends ***";
}
*/
int DQN::UpdateRandom()
{
    // Every clone_iters steps, update the clone_net_
    if (!clone_net_ || current_iteration() >= last_clone_iter_ + clone_frequency_)
    {
        LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
        CloneNet(*test_net_);
        last_clone_iter_ = current_iteration();
    }
    const auto frames_blob = net_->blob_by_name(train_frames_blob_name);
	const auto actions_blob = net_->blob_by_name(train_last_action_blob_name);
    const auto cont_blob = net_->blob_by_name(cont_blob_name);
    const auto filter_blob = net_->blob_by_name(filter_blob_name);
    const auto target_blob = net_->blob_by_name(target_blob_name);

    std::vector<float> frame_input(frame_input_size_TRAIN_, 0.0f);
	std::vector<float> actions_input(last_action_input_size_TRAIN_, 0.0f);
    std::vector<float> filter_input(filter_input_size_TRAIN_, 0.0f);
    std::vector<float> target_input(target_input_size_TRAIN_, 0.0f);
    std::vector<float> cont_input(cont_input_size_TRAIN_, 1.0f);

    for (int n = 0; n < minibatch_size_; ++n)
    {
        cont_input[cont_blob->offset(0,n,0,0)] = 0.0f;
    }

    // Randomly select unique episodes to learn from
    std::vector<int> ep_inds(replay_memory_.size());
    std::iota(ep_inds.begin(), ep_inds.end(), 0);
    auto func = [this](int i)
    {
        return random_engine() % i;
    };
    std::random_shuffle(ep_inds.begin(), ep_inds.end(), func);
    if (ep_inds.size() > minibatch_size_)
    {
        ep_inds.resize(minibatch_size_);
    }
    int batch_size = ep_inds.size();
    // Time to start each episode
    std::vector<int> ep_starts(batch_size);
    for (int n = 0; n < batch_size; ++n)
    {
        int ep_size = replay_memory_[ep_inds[n]].size();
        int last_valid_frame = ep_size - 1 - (frames_per_timestep_-1) - (unroll_-1);
        ep_starts[n] = std::uniform_int_distribution<int>
                       (0, last_valid_frame)(random_engine);
    }
    for (int u = 0; u < unroll_; ++u)
    {
        InputFramesBatch past_frames_vec;
		LastActionBatch past_action_vec;
        for (int n = 0; n < batch_size; ++n)
        {
            const Episode& episode = replay_memory_[ep_inds[n]];
            int last_frame_ts = ep_starts[n] + u + (frames_per_timestep_-1);
            CHECK_GT(episode.size(), last_frame_ts);
            if (std::get<4>(episode[last_frame_ts]))
            {
                InputFrames input_frames(frames_per_timestep_);
				LastAction input_actions(frames_per_timestep_);
                for (int i = 0; i < frames_per_timestep_; ++i)
                {
                    int ts = ep_starts[n] + u + i;
                    input_frames[i] = std::get<4>(episode[ts]).get();
					input_actions[i] = std::get<2>(episode[ts]);
                }
                past_frames_vec.push_back(input_frames);
				past_action_vec.push_back(input_actions);
            }
        }

        // Get the next state QValues
        std::vector<ActionValue> actions_and_values =
            SelectActionGreedily(*clone_net_, past_frames_vec, past_action_vec, u>0);
        // Generate the targets/filter/frames inputs
        int target_value_idx = 0;
        for (int n = 0; n < batch_size; ++n)
        {
            const Episode& episode = replay_memory_[ep_inds[n]];
            int ts = ep_starts[n] + u + (frames_per_timestep_-1);
            CHECK_GT(episode.size(), ts);
            const auto& transition = episode[ts];
            const int action = static_cast<int>(std::get<2>(transition));
            CHECK_LT(action, kOutputCount);
            const float reward = std::get<3>(transition);
            CHECK(reward >= -1.0 && reward <= 1.0);
            const float target = std::get<4>(transition) ?
                                 reward + gamma_ * actions_and_values[target_value_idx++].second :
                                 reward;
            CHECK(!std::isnan(target));
            filter_input[filter_blob->offset(u,n,action,0)] = 1;
            target_input[target_blob->offset(u,n,action,0)] = target;

			const int last_action = std::get<0>(transition);
            int last_action_idx = u + (frames_per_timestep_-1);
			actions_input[actions_blob->offset(n,last_action_idx,last_action,0)] = 1.0f;

			const auto& frame = std::get<1>(transition);
            int frame_idx = u + (frames_per_timestep_-1);
            std::copy(frame->begin(), frame->end(),
                      frame_input.begin() + frames_blob->offset(n,frame_idx,0,0));

        }
        CHECK_EQ(target_value_idx, actions_and_values.size());
    }
    // Copy in the pre-input frames
    for (int n = 0; n < batch_size; ++n)
    {
        for (int i = 0; i < frames_per_timestep_-1; ++i)
        {
            int ts = ep_starts[n] + i;
            const auto& frame = std::get<1>(replay_memory_[ep_inds[n]][ts]);
            std::copy(frame->begin(), frame->end(),
                      frame_input.begin() + frames_blob->offset(n,i,0,0));
        }
    }
    // Copy in the pre-input actions
    for (int n = 0; n < batch_size; ++n)
    {
        for (int i = 0; i < frames_per_timestep_-1; ++i)
        {
            int ts = ep_starts[n] + i;
            const Action last_action = std::get<0>(replay_memory_[ep_inds[n]][ts]);
			actions_input[actions_blob->offset(n,i,last_action,0)] = 1.0f;
        }
    }

    InputDataIntoLayers(*net_, frame_input.data(), actions_input.data(), cont_input.data(), target_input.data(), filter_input.data());
    solver_->Step(1);
    // Get loss and maybe display it
    CHECK(net_->has_blob("loss"));
    const auto loss_blob = net_->blob_by_name("loss");
    CHECK_EQ(loss_blob->count(), 1);
    float loss = loss_blob->data_at(0,0,0,0);
    CHECK(!std::isinf(loss));
    if (current_iteration() % FLAGS_loss_display_iter == 0)
    {
        LOG(INFO) << "Iteration " << current_iteration()
                  << ", loss = " << smoothed_loss_;
        smoothed_loss_ = 0;
    }
    smoothed_loss_ += loss / float(FLAGS_loss_display_iter);
    return 1;
}

void DQN::CloneNet(caffe::Net<float>& net)
{
    caffe::NetParameter net_param;
    net.ToProto(&net_param);
    net_param.mutable_state()->set_phase(net.phase());
    if (!clone_net_)
    {
        clone_net_.reset(new caffe::Net<float>(net_param));
    }
    else
    {
        clone_net_->CopyTrainedLayersFrom(net_param);
    }
}

int DQN::GetLastEpisodeSize()
{
    if (replay_memory_.empty())
    {
        return 0;
    }
    return replay_memory_.back().size();
}

void DQN::InputDataIntoLayers(caffe::Net<float>& net,
                              float* frames_input,
							  float* last_action_input,
                              float* cont_input,
                              float* target_input,
                              float* filter_input)
{
    if (frames_input != NULL)
    {
        const auto frames_input_layer =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
                net.layer_by_name(frames_layer_name));
        CHECK(frames_input_layer);
        frames_input_layer->Reset(frames_input, frames_input,
                                  frames_input_layer->batch_size());
    }
    if (last_action_input != NULL)
    {
        const auto last_action_input_layer =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
                net.layer_by_name(last_action_layer_name));
        CHECK(last_action_input_layer);
        last_action_input_layer->Reset(last_action_input, last_action_input,
                                  last_action_input_layer->batch_size());
    }
    if (cont_input != NULL)
    {
        const auto cont_input_layer =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
                net.layer_by_name(cont_layer_name));
        CHECK(cont_input_layer);
        cont_input_layer->Reset(cont_input, cont_input,
                                cont_input_layer->batch_size());
    }
    if (target_input != NULL)
    {
        const auto target_input_layer =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
                net.layer_by_name(target_layer_name));
        CHECK(target_input_layer);
        target_input_layer->Reset(target_input, target_input,
                                  target_input_layer->batch_size());
    }
    if (filter_input != NULL)
    {
        const auto filter_input_layer =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
                net.layer_by_name(filter_layer_name));
        CHECK(filter_input_layer);
        filter_input_layer->Reset(filter_input, filter_input,
                                  filter_input_layer->batch_size());
    }
}

void DQN::ClearReplayMemory()
{
    replay_memory_.clear();
    replay_memory_size_ = 0;
}

void DQN::SnapshotReplayMemory(const std::string& filename)
{
    std::ofstream ofile(filename.c_str(), std::ios_base::out | std::ofstream::binary);
    boost::iostreams::filtering_ostream out;
    out.push(boost::iostreams::gzip_compressor());
    out.push(ofile);
    int num_episodes = replay_memory_.size();
    out.write((char*)&num_episodes, sizeof(int));
    for (const Episode& ep : replay_memory_)
    {
        int ep_len = ep.size();
        out.write((char*)&ep_len, sizeof(int));
    }
    for (const Episode& ep : replay_memory_)
    {
        for (const Transition& t : ep)
        {
            const Action& last_action = std::get<0>(t);
            out.write((char*)&last_action, sizeof(Action));
            const FrameDataSp& frame = std::get<1>(t);
            out.write((char*)frame->begin(), kCroppedFrameDataSize * sizeof(uint8_t));
            const Action& action = std::get<2>(t);
            out.write((char*)&action, sizeof(Action));
            const float& reward = std::get<3>(t);
            out.write((char*)&reward, sizeof(float));
        }
    }
    LOG(INFO) << "Saved memory of size " << replay_memory_size_;
}


void DQN::LoadReplayMemory(const std::string& filename)
{
    LOG(INFO) << "Loading memory from " << filename;
    ClearReplayMemory();
    std::ifstream ifile(filename.c_str(), std::ios_base::in | std::ofstream::binary);
    boost::iostreams::filtering_istream in;
    in.push(boost::iostreams::gzip_decompressor());
    in.push(ifile);
    int num_episodes;
    in.read((char*)&num_episodes, sizeof(int));
    replay_memory_.resize(num_episodes);
    for (int i = 0; i < num_episodes; ++i)
    {
        int ep_len;
        in.read((char*)&ep_len, sizeof(int));
        replay_memory_[i].resize(ep_len);
        replay_memory_size_ += ep_len;
    }
    for (int i = 0; i < num_episodes; ++i)
    {
        Episode& ep = replay_memory_[i];
        for (int j = 0; j < ep.size(); ++j)
        {
			Transition& t = ep[j];
			in.read((char*)&std::get<0>(t), sizeof(Action));

            FrameDataSp frame = std::make_shared<FrameData>();
            in.read((char*)frame->begin(), kCroppedFrameDataSize * sizeof(uint8_t));
            std::get<1>(t) = frame;
            if (j > 0)
            {
                std::get<4>(ep[j-1]) = frame;
            }
            in.read((char*)&std::get<2>(t), sizeof(Action));
            in.read((char*)&std::get<3>(t), sizeof(float));
            std::get<4>(t) == boost::none;
        }
    }
    LOG(INFO) << "replay_mem_size = " << replay_memory_size_;
}

} // namespace dqn
