syntax = "proto2";

message Request {
	enum Type {
		PING        = 0;
		SMT_COMMAND = 1;
		CLIENT_QUIT = 2;
	}
	required Type   type        = 1;
	optional string smt_command = 2;
}

message Reply {
	enum Type {
		PONG      = 0;
		SMT_REPLY = 1;
		ERROR     = 2;
	}
	required Type   type      = 1;
	required int32  status    = 2;
	optional string error_msg = 3;
	optional string smt_reply = 4;
	optional string smt_err   = 5;
}

message Smlp {
	required int32 version  = 1;

	/* fresh for requests
	 * requests's msg_id for replies */
	required int32 msg_id   = 2;

	/* setting none or both is invalid */
	optional Request request = 4;
	optional Reply   reply   = 5;
}
