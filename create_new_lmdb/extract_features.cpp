#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include <fstream>  
#include <iostream>  
//#define OUTPUT

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;

using namespace std;
namespace db = caffe::db;

#define NUM_REQUIRED_ARGS 8//表示的是该函数最低要输入多少个参数

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
	return feature_extraction_pipeline<float>(argc, argv);
	//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	const int num_required_args = NUM_REQUIRED_ARGS;
	if (argc < num_required_args) {
		LOG(ERROR) <<
			"This program takes in a trained network and an input data layer, and then"
			" extract features of the input data produced by the net.\n"
			"Usage: extract_features  pretrained_net_param"
			"  feature_extraction_proto_file image_LMDB_file  extract_feature_blob_name1[,name2,...]"
			"  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
			"  [CPU/GPU] [DEVICE_ID=0]\n"
			"Note: you can extract multiple features in one pass by specifying"
			" multiple feature blob names and dataset names separated by ','."
			" The names cannot contain white space characters and the number of blobs"
			" and datasets must be equal.";
		return 1;
	}
	int arg_pos = num_required_args;

	arg_pos = num_required_args;
	if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
		LOG(ERROR) << "Using GPU";
		int device_id = 0;
		if (argc > arg_pos + 1) {
			device_id = atoi(argv[arg_pos + 1]);
			CHECK_GE(device_id, 0);
		}
		LOG(ERROR) << "Using Device_id=" << device_id;
		Caffe::SetDevice(device_id);
		Caffe::set_mode(Caffe::GPU);
	}
	else {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}

	arg_pos = 0;  // the name of the executable
	std::string pretrained_binary_proto(argv[++arg_pos]);

	// Expected prototxt contains at least one data layer such as
	//  the layer data_layer_name and one feature blob such as the
	//  fc7 top blob to extract features.
	/*
	layers {
	name: "data_layer_name"
	type: DATA
	data_param {
	source: "/path/to/your/images/to/extract/feature/images_leveldb"
	mean_file: "/path/to/your/image_mean.binaryproto"
	batch_size: 128
	crop_size: 227
	mirror: false
	}
	top: "data_blob_name"
	top: "label_blob_name"
	}
	layers {
	name: "drop7"
	type: DROPOUT
	dropout_param {
	dropout_ratio: 0.5
	}
	bottom: "fc7"
	top: "fc7"
	}
	*/
	std::string feature_extraction_proto(argv[++arg_pos]);
	boost::shared_ptr<Net<Dtype> > feature_extraction_net(
		new Net<Dtype>(feature_extraction_proto, caffe::TEST));
	feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

	//此处为增加的输入变量，为输入的命令参数的第四个变量
	std::string image_LMDB_file(argv[++arg_pos]);

	std::string extract_feature_blob_names(argv[++arg_pos]);
	std::vector<std::string> blob_names;
	boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

	std::string save_feature_dataset_names(argv[++arg_pos]);
	std::vector<std::string> dataset_names;
	boost::split(dataset_names, save_feature_dataset_names,
		boost::is_any_of(","));
	CHECK_EQ(blob_names.size(), dataset_names.size()) <<
		" the number of blob names and dataset names must be equal";
	size_t num_features = blob_names.size();

	for (size_t i = 0; i < num_features; i++) {
		CHECK(feature_extraction_net->has_blob(blob_names[i]))
			<< "Unknown feature blob name " << blob_names[i]
			<< " in the network " << feature_extraction_proto;
	}

	int num_mini_batches = atoi(argv[++arg_pos]);

	std::vector<boost::shared_ptr<db::DB> > feature_dbs;
	std::vector<boost::shared_ptr<db::Transaction> > txns;

	//此处增加对LMDB数据库的读取操作，定义了数据库读取和游标
	boost::shared_ptr<db::DB> imageData_dbs;
	boost::shared_ptr<db::Cursor> txnsImageData;
	imageData_dbs.reset(db::GetDB("lmdb"));
	//打开指定的数据库，并初始化游标
	imageData_dbs->Open(image_LMDB_file, db::READ);
	txnsImageData.reset(imageData_dbs->NewCursor());

	const char* db_type = argv[++arg_pos];
	for (size_t i = 0; i < num_features; ++i) {
		string ss = dataset_names[i];
		string sss = dataset_names.at(i);

		LOG(INFO) << "Opening dataset " << dataset_names[i];
		boost::shared_ptr<db::DB> db(db::GetDB(db_type));
		db->Open(dataset_names.at(i), db::NEW);
		feature_dbs.push_back(db);
		boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
		txns.push_back(txn);
	}

	LOG(ERROR) << "Extracting Features";

#ifdef OUTPUT
	ofstream outfile_label("E:/out_label.txt", ofstream::app);  //想指定文件中写入信息
	ofstream outfile_feature_label("E:/out_feature_label.txt", ofstream::app);  //想指定文件中写入信息
	ofstream outfile_imageData("E:/out_imageData.txt", ofstream::app);  //想指定文件中写入信息
#endif
	
	Datum datum;
	std::vector<int> image_indices(num_features, 0);
	for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {

		feature_extraction_net->Forward();
		for (int i = 0; i < num_features; ++i) {
			const boost::shared_ptr<Blob<Dtype> > feature_blob =
				feature_extraction_net->blob_by_name(blob_names[i]);
			int batch_size = feature_blob->num();
			int dim_features = feature_blob->count() / batch_size; 
			const Dtype* feature_blob_data;

			//将此游标指向该数据库的第一条记录
			txnsImageData->SeekToFirst();
			//此处将数据库的游标指向当前batch的首位置
			for (int k = 0; k < batch_index * batch_size; ++k)
			{
				txnsImageData->Next();
				if (!txnsImageData->valid()) {
					DLOG(INFO) << "Restarting data prefetching from start.";
					txnsImageData->SeekToFirst(); 
				}
			}

			for (int n = 0; n < batch_size; ++n) 
			{
				//对datum中的数据进行清零操作
				datum.clear_data();
				datum.clear_float_data();
				datum.clear_label();
				datum.clear_width();
				datum.clear_height();

				//对当前游标指向的行数据txnsImageData->value()进行反序列化操作，并将其数据保存在datum的特定位置处
				datum.ParseFromString(txnsImageData->value());//对当前游标指向的数据进行反序列化，并将反序列化之后的数据存到datum结构体中。

#ifdef OUTPUT
				outfile_label << (float)(datum.label()) << "   ";
				cout << (float)(datum.label()) << "  " << endl;
				outfile_imageData << "开始写入第" << batch_index * batch_size + n << "个图像的数据：共有imageData " << datum.width() * datum.height()  << "个。" << endl;

				for (int ii = 0; ii < datum.width() * datum.height();ii++)
				{
					outfile_imageData << (int)(datum.data().data()[ii]) << "  ";
				}

				outfile_imageData << "写入第" << batch_index * batch_size + n << "个图像的数据结束" << endl;
#endif

				//将获取到的feture_blob_data写入到Datum的float_data中
#ifdef OUTPUT
				outfile_feature_label << "开始写入第" << batch_index * batch_size + n << "个图像的数据：共有feature "<< dim_features<< "维。" << endl;
#endif
				feature_blob_data = feature_blob->cpu_data() +
					feature_blob->offset(n);
				for (int s = 0; s < dim_features; ++s)
				{
					datum.add_float_data((float)(feature_blob_data[s]));//该处是将提取到的feature当作当前图片的lebels的值存入datum结构体中。此处的结构体是进行了相应的改进，该Datum结构体中的label是存储的一个float型的数组
#ifdef OUTPUT
					outfile_feature_label << (float)(datum.float_data().data()[s])<< "  ";
#endif
				}
#ifdef OUTPUT
				outfile_feature_label << "写入第" << batch_index * batch_size + n << "个图像的数据结束" << endl;
				
				cout << "执行第 " << batch_index * batch_size + n << "个图像的操作" << endl;
#endif
				string key_str = caffe::format_int(image_indices[i], 10);

				string out;
				CHECK(datum.SerializeToString(&out));
				txns.at(i)->Put(key_str, out);
				++image_indices[i];
				if (image_indices[i] % 1000 == 0) {
					txns.at(i)->Commit();
					txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
					LOG(ERROR) << "Extracted features of " << image_indices[i] <<
						" query images for feature blob " << blob_names[i];
				}

				//将游标指向下一行记录的首位置
				txnsImageData->Next();
			}  // for (int n = 0; n < batch_size; ++n)
		}  // for (int i = 0; i < num_features; ++i)
	}  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
	// write the last batch
	for (int i = 0; i < num_features; ++i) {
		if (image_indices[i] % 1000 != 0) {
			txns.at(i)->Commit();
		}
		LOG(ERROR) << "Extracted features of " << image_indices[i] <<
			" query images for feature blob " << blob_names[i];
		feature_dbs.at(i)->Close();
#ifdef OUTPUT
		outfile_label.close();
		outfile_feature_label.close();
		outfile_imageData.close();
#endif
	}

	LOG(ERROR) << "Successfully extracted the features!";
	return 0;
}
