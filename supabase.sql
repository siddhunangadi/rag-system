create extension if not exists vector;

create table if not exists policy_documents (
  id uuid primary key default gen_random_uuid(),
  name text unique not null,
  object_path text unique not null,
  created_at timestamptz not null default now()
);

create table if not exists policy_chunks (
  chunk_id text primary key,
  document_id uuid not null references policy_documents(id) on delete cascade,
  document text not null,
  page integer not null,
  text text not null,
  embedding vector(384) not null
);

create index if not exists policy_chunks_embedding_idx
  on policy_chunks using hnsw (embedding vector_cosine_ops);

insert into storage.buckets (id, name, public) values ('policy-pdfs', 'policy-pdfs', false)
on conflict (id) do nothing;

create or replace function replace_policy_document(p_name text, p_path text, p_chunks jsonb)
returns void language plpgsql security definer set search_path = public as $$
declare p_id uuid;
begin
  delete from policy_documents where name = p_name;
  insert into policy_documents (name, object_path) values (p_name, p_path) returning id into p_id;
  insert into policy_chunks (chunk_id, document_id, document, page, text, embedding)
  select value->>'chunk_id', p_id, value->>'document', (value->>'page')::integer,
         value->>'text', (value->>'embedding')::vector
  from jsonb_array_elements(p_chunks);
end $$;

create or replace function match_policy_chunks(query_embedding vector(384), match_count integer)
returns table(chunk_id text, similarity real) language sql stable set search_path = public as $$
  select chunk_id, (1 - (embedding <=> query_embedding))::real
  from policy_chunks order by embedding <=> query_embedding limit match_count
$$;

revoke all on policy_documents, policy_chunks from anon, authenticated;
revoke all on function replace_policy_document(text, text, jsonb) from public;
revoke all on function match_policy_chunks(vector, integer) from public;
grant execute on function replace_policy_document(text, text, jsonb) to service_role;
grant execute on function match_policy_chunks(vector, integer) to service_role;
